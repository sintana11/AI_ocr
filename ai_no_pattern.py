from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
from collections import Counter, defaultdict
import cv2
import numpy as np
import re
import logging
from typing import Optional, Dict, List, Tuple
import time
import json
import os
import gc

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    MODEL_PATH = os.environ.get("MODEL_PATH", "D:/gog/AI_ocr/model/V11n/weights/best.pt")
    PORT = int(os.environ.get("PORT", 8000))
    YOLO_CONF = 0.25
    YOLO_IMGSZ = 640
    DENOISE_H = 20
    ADAPTIVE_THRESH_BLOCK = 31
    ADAPTIVE_THRESH_C = 10
    OCR_TOP_CROP_RATIO = 0.45
    MIN_OCR_DIM = 300
    MAX_INPUT_SIZE = 1280
    LABEL_NOISE_WORDS = {
        "HEALTH", "BIOMEDICAL", "ENGINEERING", "SERVICE", "MEDIUM",
        "RISK", "DEVICE", "CALIBRATION", "DATE", "MODEL", "SERIAL",
        "NHEALTH", "PAT", "NEXT", "DUE", "WARRANTY", "BRAND",
    }



model = YOLO(Config.MODEL_PATH)
# ถ้ารันใน Docker (CPU-only) ให้ตั้ง EASYOCR_GPU=false
_use_gpu = os.environ.get("EASYOCR_GPU", "true").lower() == "true"
reader = easyocr.Reader(['en'], gpu=_use_gpu, verbose=False)
app = Flask(__name__)


def limit_image_size(img: np.ndarray, max_size: int = None) -> np.ndarray:
    """ย่อภาพถ้าด้านยาวสุดเกิน max_size — ลด RAM ทั้ง pipeline"""
    if max_size is None:
        max_size = Config.MAX_INPUT_SIZE
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(f"Resize input: {w}x{h} -> {new_w}x{new_h} (save RAM)")
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_resize_scale(crop: np.ndarray) -> float:
    """คำนวณ scale อัตโนมัติตามขนาด crop — crop เล็กขยายมาก, crop ใหญ่ขยายน้อย"""
    min_dim = min(crop.shape[0], crop.shape[1])
    if min_dim >= Config.MIN_OCR_DIM:
        return 1.0  # ใหญ่พอแล้ว ไม่ต้องขยาย
    scale = Config.MIN_OCR_DIM / min_dim
    scale = min(scale, 4.0)  # จำกัดไม่เกิน 4x เพื่อไม่กิน RAM เกิน
    return round(scale, 2)


# ==============================
# IMAGE VALIDATION
# ==============================
def validate_image(img: np.ndarray) -> bool:
    return (
        img is not None and
        img.size > 0 and
        len(img.shape) >= 2 and
        img.shape[0] >= 10 and
        img.shape[1] >= 10
    )


# ==============================
# PREPROCESSING STRATEGIES (ใช้ adaptive resize)
# ==============================

def preprocess_v1_clahe(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 1: CLAHE (fast, good contrast)"""
    try:
        scale = compute_resize_scale(img)
        if scale > 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception as e:
        logger.warning(f"[v1] failed: {e}")
        return None


def preprocess_v2_adaptive_thresh(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        scale = compute_resize_scale(img)
        if scale > 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            Config.ADAPTIVE_THRESH_BLOCK,
            Config.ADAPTIVE_THRESH_C
        )
        return thresh
    except Exception as e:
        logger.warning(f"[v2] failed: {e}")
        return None


def preprocess_v3_otsu(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        scale = compute_resize_scale(img)
        if scale > 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    except Exception as e:
        logger.warning(f"[v3] failed: {e}")
        return None


def preprocess_v4_sharpened(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE * 1.5, fy=Config.RESIZE_SCALE * 1.5,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return clahe.apply(sharpened)
    except Exception as e:
        logger.warning(f"[v4] failed: {e}")
        return None


def preprocess_v5_inverted(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE,
                         interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(inverted)
    except Exception as e:
        logger.warning(f"[v5] failed: {e}")
        return None


def preprocess_v6_morph(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.dilate(gray, kernel, iterations=1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        logger.warning(f"[v6] failed: {e}")
        return None


def preprocess_v7_bilateral(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception as e:
        logger.warning(f"[v7] failed: {e}")
        return None


def preprocess_v8_highres(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        img = cv2.resize(img, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        logger.warning(f"[v8] failed: {e}")
        return None


ALL_PREPROCESS_STRATEGIES = [
    ("v1_clahe", preprocess_v1_clahe),
    ("v2_adaptive", preprocess_v2_adaptive_thresh),
    ("v3_otsu", preprocess_v3_otsu),
    ("v4_sharp", preprocess_v4_sharpened),
    ("v5_inverted", preprocess_v5_inverted),
    ("v6_morph", preprocess_v6_morph),
    ("v7_bilateral", preprocess_v7_bilateral),
    ("v8_highres", preprocess_v8_highres),
]


# ==============================
# OCR TEXT CLEANER (ไม่กำหนด pattern)
# ==============================

def clean_ocr_text(text: str) -> str:
    """ทำความสะอาด text เบื้องต้น"""
    text = text.strip()
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'[^\w\s\-./]', '', text)
    return text.strip()


def _is_label_noise(text: str) -> bool:
    """ตรวจว่า text เป็นคำบนป้ายที่ไม่ใช่รหัสครุภัณฑ์ (fuzzy matching)"""
    word = text.upper().strip()
    alpha_part = re.sub(r'[^A-Z]', '', word)

    # exact match
    for noise in Config.LABEL_NOISE_WORDS:
        if noise in word:
            return True

    # partial match: ถ้าส่วนตัวอักษร >= 3 ตัว ตรงกับ noise word
    if len(alpha_part) >= 3:
        for noise in Config.LABEL_NOISE_WORDS:
            if alpha_part in noise or noise in alpha_part:
                return True

    # ตัวอักษรล้วน (ไม่มีตัวเลข) >= 3 → มักเป็นคำบนป้าย
    if alpha_part == word and len(word) >= 3:
        return True

    return False


def extract_text_no_pattern(text: str) -> Optional[Tuple[str, float]]:
    """
    ดึง text จาก OCR โดยไม่กำหนด pattern ตายตัว
    - ต้องมีทั้งตัวอักษรและตัวเลข
    - ต้องมีตัวเลขอย่างน้อย 3 ตัว (รหัสจริงมีตัวเลขเยอะ เช่น 04622, 00248)
    - กรอง noise words
    """
    cleaned = clean_ocr_text(text)
    if not cleaned or len(cleaned) < 4:
        return None

    upper = cleaned.upper()

    if _is_label_noise(upper):
        return None

    alnum = re.sub(r'[^A-Z0-9\-]', '', upper)
    if len(alnum) < 4:
        return None

    has_letter = bool(re.search(r'[A-Z]', alnum))
    digits = re.findall(r'[0-9]', alnum)
    if not has_letter or len(digits) < 3:
        return None

    score = min(1.0, len(alnum) / 8.0)
    return alnum, score


# ==============================
# GENERAL OCR CORRECTION (ไม่ hardcode pattern ใดๆ)
# ==============================

DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E',
    '4': 'A', '5': 'S', '6': 'G', '8': 'S', '9': 'S'
}
LETTER_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2',
    'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0', 'C': '0'
}


def correct_ocr_by_context(text: str) -> str:
    """
    แก้ไข OCR misread โดยใช้ context ของตำแหน่ง (general — ไม่ hardcode pattern)

    หลักการ: รหัสครุภัณฑ์ = prefix ตัวอักษร + suffix ตัวเลข
    - ตำแหน่ง prefix: ตัวเลข → ตัวอักษร (เช่น 8→S, 9→S)
    - ตำแหน่ง suffix: ตัวอักษร → ตัวเลข (เช่น O→0, S→5)

    ตัวอย่าง: 89H04622 → SSH04622
    """
    text = text.upper().replace('-', '')
    if len(text) < 3:
        return text

    # หาจุดแบ่ง prefix/suffix
    split_pos = 0
    for i in range(len(text)):
        if text[i].isalpha():
            rest = text[i + 1:]
            digit_count = sum(1 for c in rest if c.isdigit())
            if rest and digit_count / len(rest) >= 0.7:
                split_pos = i + 1
                break

    if split_pos == 0:
        for i in range(len(text)):
            if text[i].isdigit():
                rest = text[i:]
                digit_count = sum(1 for c in rest if c.isdigit())
                if digit_count / len(rest) >= 0.8:
                    split_pos = i
                    break

    if split_pos == 0:
        return text

    prefix = text[:split_pos]
    suffix = text[split_pos:]

    # แก้ prefix: ตัวเลข → ตัวอักษร
    corrected_prefix = ''
    for c in prefix:
        if c.isdigit() and c in DIGIT_TO_LETTER:
            corrected_prefix += DIGIT_TO_LETTER[c]
        else:
            corrected_prefix += c

    # แก้ suffix: ตัวอักษร → ตัวเลข
    corrected_suffix = ''
    for c in suffix:
        if c.isalpha() and c in LETTER_TO_DIGIT:
            corrected_suffix += LETTER_TO_DIGIT[c]
        else:
            corrected_suffix += c

    result = corrected_prefix + corrected_suffix
    if result != text:
        logger.info(f"🔧 OCR correction: '{text}' → '{result}' (split at {split_pos})")

    return result


def build_positional_consensus(all_candidates: List[Tuple[str, float, str]]) -> Optional[str]:
    """
    สร้าง consensus code จาก candidates โดย vote ทีละตำแหน่ง
    """
    if not all_candidates:
        return None

    lengths = Counter(len(code) for code, _, _ in all_candidates)
    target_len = lengths.most_common(1)[0][0]

    same_len = [(code, conf) for code, conf, _ in all_candidates if len(code) == target_len]
    if not same_len:
        return None

    consensus = []
    for i in range(target_len):
        char_votes = Counter()
        for code, conf in same_len:
            char_votes[code[i]] += conf
        best_char = char_votes.most_common(1)[0][0]
        consensus.append(best_char)

    return ''.join(consensus)


# ==============================
# OCR FUNCTIONS
# ==============================

def run_easyocr(img_gray: np.ndarray, top_crop: bool = True) -> List[Tuple[str, float]]:
    """Run EasyOCR"""
    try:
        h = img_gray.shape[0]
        img = img_gray[int(h * Config.OCR_TOP_CROP_RATIO):, :] if top_crop else img_gray

        results = reader.readtext(
            img,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            paragraph=False,
            detail=1,
            contrast_ths=0.3,
            adjust_contrast=0.7,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.3,
            mag_ratio=1.0,  # ลดจาก 1.5 — ภาพถูก resize 2x แล้ว ไม่ต้องขยายซ้ำ
            width_ths=0.7,
        )
        return [(text, conf) for (_, text, conf) in results]
    except Exception as e:
        logger.warning(f"EasyOCR error: {e}")
        return []


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)


def try_all_ocr(processed_img: np.ndarray) -> List[Tuple[str, float, str]]:
    """ลอง EasyOCR + rotation — เก็บทุก text ที่อ่านได้"""
    candidates = []

    images_to_try = [
        ("", processed_img),
        ("_rot2", rotate_image(processed_img, 2)),
        ("_rot-2", rotate_image(processed_img, -2)),
    ]

    for suffix, img in images_to_try:
        for text, conf in run_easyocr(img, top_crop=True):
            result = extract_text_no_pattern(text)
            if result:
                clean_text, score = result
                candidates.append((clean_text, conf * score, f"easyocr_crop{suffix}"))

        for text, conf in run_easyocr(img, top_crop=False):
            result = extract_text_no_pattern(text)
            if result:
                clean_text, score = result
                candidates.append((clean_text, conf * score, f"easyocr_full{suffix}"))

    return candidates


def correct_ocr_format(text: str) -> str:
    """
    แก้ปัญหา OCR อ่านผิดทั้งส่วนหัว (ตัวอักษร) และส่วนท้าย (ตัวเลข)
    อย่างปลอดภัย:
    - 3 ตัวแรก: ควรเป็นตัวอักษร (แก้ 8->S, 5->S)
    - ที่เหลือ: ควรเป็นตัวเลข (แก้ O->0, I->1, S->5, Z->2)
    """
    if len(text) < 5:
        return text

    # 0. Fix common multi-character misreads (เช่น H มักโดนอ่านเป็น F1 หรือ FI)
    text = text.replace('F1', 'H').replace('FI', 'H')

    prefix = text[:3].upper()
    suffix = text[3:].upper()

    # 1. แก้ส่วนท้าย (Suffix): ส่วนท้ายรหัสครุภัณฑ์ควรเป็นตัวเลข
    # ถ้ามีตัวอักษรที่หน้าตาคล้ายเลขหลงมา ให้แปลงเป็นเลขให้หมด
    suffix_fixes = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'C': '0', 'Q': '0', 'D': '0'}
    fixed_suffix = ""
    for char in suffix:
        if char in suffix_fixes:
            fixed_suffix += suffix_fixes[char]
        else:
            fixed_suffix += char

    # 2. แก้ส่วนหน้า (Prefix): 3 ตัวแรกควรเป็นอักษร
    # ถ้าด้านหลังกลายเป็น(หรือเป็น)เลขเกือบหมด -> ข้างหน้าควรแก้เลขเป็นอักษร
    digits_in_suffix = sum(1 for c in fixed_suffix if c.isdigit())
    if len(fixed_suffix) > 0 and (digits_in_suffix / len(fixed_suffix)) >= 0.7:
        prefix_fixes = {'8': 'S', '9': 'S', '5': 'S', '0': 'O', '1': 'I', '2': 'Z'}
        fixed_prefix = ""
        for char in prefix:
            if char in prefix_fixes:
                fixed_prefix += prefix_fixes[char]
            else:
                fixed_prefix += char
        
        corrected = fixed_prefix + fixed_suffix
        if corrected != text:
             logger.info(f"🔧 OCR Fix: '{text}' → '{corrected}'")
        return corrected

    # ถ้าเงื่อนไขไม่ตรง ก็ส่งส่วนท้ายที่แก้แล้วประกบกลับ
    corrected = prefix + fixed_suffix
    if corrected != text:
         logger.info(f"🔧 OCR Suffix Fix: '{text}' → '{corrected}'")
    
    return corrected


def pick_best_candidate(all_candidates: List[Tuple[str, float, str]]) -> Optional[Tuple[str, float, str]]:
    """
    เลือก text ที่ดีที่สุดจาก candidates ทั้งหมด
    
    ใช้ VOTE-FIRST strategy:
    1. นับ votes ของแต่ละ text
    2. เลือก text ที่ votes มากที่สุด
    3. ส่งผ่าน correct_ocr_format เพื่อแก้ให้ถูกต้องที่สุด
    """
    if not all_candidates:
        return None

    text_info = defaultdict(lambda: {"count": 0, "best_conf": 0.0, "best_engine": ""})
    for text, conf, engine in all_candidates:
        info = text_info[text]
        info["count"] += 1
        if conf > info["best_conf"]:
            info["best_conf"] = conf
            info["best_engine"] = engine

    sorted_texts = sorted(
        text_info.items(),
        key=lambda x: (x[1]["count"], x[1]["best_conf"]),
        reverse=True
    )

    best_text = sorted_texts[0][0]
    best_info = sorted_texts[0][1]

    # แก้ไขทั้ง Prefix และ Suffix ให้ตรงกับฟอร์แมตครุภัณฑ์ (เช่น 89HO3558 -> SSH03558)
    final_text = correct_ocr_format(best_text)

    total = len(all_candidates)
    logger.info(
        f"📊 Voting: top5={[(t, i['count']) for t, i in sorted_texts[:5]]} "
        f"→ raw='{best_text}' → final='{final_text}' (votes={best_info['count']}/{total}, conf={best_info['best_conf']:.3f})"
    )

    return final_text, best_info["best_conf"], best_info["best_engine"]


# ==============================
# MAIN DETECTION PIPELINE
# ==============================

def detect_code(img: np.ndarray) -> Dict:
    """ตรวจจับและอ่านรหัสจากภาพ — adaptive resize + ประหยัดทรัพยากร"""
    start = time.time()
    detected = False

    # จำกัดขนาดภาพอินพุตเพื่อลด RAM ตลอด pipeline
    img = limit_image_size(img)

    # YOLO detect
    results = model.predict(
        source=img,
        conf=0.25,
        imgsz=Config.YOLO_IMGSZ,
        verbose=False,
        max_det=1,
        device='cpu',
        half=False
    )

    # ดึง boxes ออกมาแล้วปล่อย YOLO results เพื่อคืน memory
    boxes_data = []
    for r in results:
        if r.boxes:
            detected = True
            for box in sorted(r.boxes, key=lambda x: x.conf[0], reverse=True)[:1]:
                boxes_data.append({
                    'xyxy': list(map(int, box.xyxy[0])),
                    'conf': float(box.conf[0])
                })
    del results

    FAST_STRATEGIES = ["v1_clahe", "v3_otsu", "v2_adaptive"]
    strategy_dict = dict(ALL_PREPROCESS_STRATEGIES)

    for box_info in boxes_data:
        x1, y1, x2, y2 = box_info['xyxy']
        box_conf = box_info['conf']

        # เพิ่ม padding 20% (จากเดิม 10%) เพื่อให้ OCR มี context มากขึ้น
        pad_x = int((x2 - x1) * 0.20)
        pad_y = int((y2 - y1) * 0.20)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img.shape[1], x2 + pad_x)
        y2 = min(img.shape[0], y2 + pad_y)

        crop = img[y1:y2, x1:x2].copy()
        if not validate_image(crop):
            continue

        adaptive_scale = compute_resize_scale(crop)
        logger.info(f"🔍 crop: {crop.shape} | yolo_conf: {box_conf:.2f} | adaptive_scale: {adaptive_scale}x")

        all_candidates = []
        best_conf = 0.0

        # === PASS 1: full image (ไม่ crop top) ===
        for strategy_name in FAST_STRATEGIES:
            processed = strategy_dict[strategy_name](crop)
            if processed is None:
                continue

            for text, conf in run_easyocr(processed, top_crop=False):
                result = extract_text_no_pattern(text)
                if result:
                    clean_text, score = result
                    final_conf = conf * score
                    all_candidates.append((clean_text, final_conf, f"easyocr_full+{strategy_name}"))
                    if final_conf > best_conf:
                        best_conf = final_conf

            del processed

            if all_candidates and best_conf >= 0.35:
                break

        # === PASS 2: ถ้ายังไม่เจอ ลอง top_crop (ตัดครึ่งบน ดูแค่ส่วนล่างที่มีรหัส) ===
        if not all_candidates or best_conf < 0.25:
            for strategy_name in FAST_STRATEGIES[:2]:  # ลองแค่ 2 strategy
                processed = strategy_dict[strategy_name](crop)
                if processed is None:
                    continue

                for text, conf in run_easyocr(processed, top_crop=True):
                    result = extract_text_no_pattern(text)
                    if result:
                        clean_text, score = result
                        final_conf = conf * score
                        all_candidates.append((clean_text, final_conf, f"easyocr_crop+{strategy_name}"))
                        if final_conf > best_conf:
                            best_conf = final_conf

                del processed

                if all_candidates and best_conf >= 0.35:
                    break

        del crop

        # เลือกผลลัพธ์ที่ดีที่สุด
        if all_candidates:
            best = pick_best_candidate(all_candidates)
            if best:
                text, confidence, engine = best
                if confidence >= 0.20:
                    elapsed = time.time() - start
                    logger.info(f"⚡ SUCCESS | text={text} | conf={confidence:.3f} | engine={engine} | time={elapsed:.2f}s")
                    return {
                        "status": "success",
                        "message": "Detect และ OCR สำเร็จ",
                        "code": text,
                        "source": engine,
                        "confidence": box_conf,
                        "ocr_confidence": round(confidence, 3),
                        "total_candidates": len(all_candidates),
                        "processing_time": f"{elapsed:.2f}s"
                    }
                else:
                    logger.warning(f"⚠️ OCR conf {confidence:.3f} < 0.20, rejected '{text}'")

        logger.warning("⚠️ All strategies failed for this box")

    if detected:
        return {
            "status": "detected_no_ocr",
            "message": "Detect เจอป้าย แต่ OCR อ่านข้อความไม่ได้เลย",
            "processing_time": f"{time.time() - start:.2f}s"
        }

    return {
        "status": "not_detected",
        "message": "YOLO ไม่พบป้าย",
        "processing_time": f"{time.time() - start:.2f}s"
    }





# ==============================
# API
# ==============================

@app.route("/ocr", methods=["POST"])
def ocr_image():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    img = cv2.imdecode(
        np.frombuffer(request.files["file"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    if not validate_image(img):
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    result = detect_code(img)
    del img  # ปล่อยภาพต้นฉบับคืน RAM
    gc.collect()  # บังคับคืน memory ทันที
    return jsonify(result)


# ==============================
# DEBUG ENDPOINT
# ==============================

@app.route("/debug", methods=["POST"])
def debug_image():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    img = cv2.imdecode(
        np.frombuffer(request.files["file"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    if not validate_image(img):
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    debug_results = []
    for strategy_name, preprocess_fn in ALL_PREPROCESS_STRATEGIES:
        processed = preprocess_fn(img)
        if processed is None:
            debug_results.append({"strategy": strategy_name, "result": "preprocess_failed"})
            continue

        all_texts = [(t, c, "easyocr") for t, c in run_easyocr(processed, top_crop=False)]
        found_list = []
        for text, conf, engine in all_texts:
            result = extract_text_no_pattern(text)
            if result:
                found_list.append({"text": result[0], "score": result[1], "engine": engine, "raw_text": text})

        debug_results.append({
            "strategy": strategy_name,
            "raw_texts": [(t, round(c, 3)) for t, c, _ in all_texts[:10]],
            "found": found_list
        })

    return jsonify({"debug": debug_results})


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    logger.info(f"🚀 Starting OCR API (No Pattern) on port {Config.PORT}")
    app.run(host="0.0.0.0", port=Config.PORT, debug=False)
