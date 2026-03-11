from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
import logging
from typing import Optional, Dict, List, Tuple
import time
import json
import os


# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    MODEL_PATH = "D:/gog/V11newest1/weights/best.pt"
    PORT = 8000
    OUTPUT_JSON_DIR = "json_results"
    YOLO_CONF = 0.3
    YOLO_IMGSZ = 640
    RESIZE_SCALE = 2.0
    DENOISE_H = 20
    ADAPTIVE_THRESH_BLOCK = 31
    ADAPTIVE_THRESH_C = 10
    OCR_TOP_CROP_RATIO = 0.45
    SSH_PATTERN = re.compile(r'SSH\s*\d{3,}', re.IGNORECASE)
    SSH_LOOSE_PATTERN = re.compile(r'[S5][S5H][SH0-9][\s\-_]*\d{2,}', re.IGNORECASE)
    GENERAL_CODE_PATTERN = re.compile(r'[A-Z]{2,5}[\s\-_]*\d{2,}', re.IGNORECASE)
    LABEL_NOISE_WORDS = {
        "HEALTH", "BIOMEDICAL", "ENGINEERING", "SERVICE", "MEDIUM",
        "RISK", "DEVICE", "CALIBRATION", "DATE", "MODEL", "SERIAL",
        "NHEALTH", "PAT", "NEXT", "DUE", "WARRANTY", "BRAND",
    }


os.makedirs(Config.OUTPUT_JSON_DIR, exist_ok=True)
model = YOLO(Config.MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True, verbose=False)
app = Flask(__name__)


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
# PREPROCESSING STRATEGIES
# ==============================

def preprocess_v1_clahe(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 1: CLAHE (original approach, fast)"""
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE,
                         interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception as e:
        logger.warning(f"[v1] failed: {e}")
        return None


def preprocess_v2_adaptive_thresh(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 2: Adaptive Threshold - ดีสำหรับรูปที่มีแสงไม่สม่ำเสมอ"""
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # denoise ก่อน
        gray = cv2.fastNlMeansDenoising(gray, h=15)
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
    """Strategy 3: Otsu Binarization - ดีสำหรับภาพที่มี contrast ชัดเจน"""
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    except Exception as e:
        logger.warning(f"[v3] failed: {e}")
        return None


def preprocess_v4_sharpened(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 4: Sharpen + CLAHE - ดีสำหรับภาพเบลอ"""
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE * 1.5, fy=Config.RESIZE_SCALE * 1.5,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Unsharp masking
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return clahe.apply(sharpened)
    except Exception as e:
        logger.warning(f"[v4] failed: {e}")
        return None


def preprocess_v5_inverted(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 5: Inverted - ดีสำหรับตัวอักษรสีอ่อนบนพื้นมืด"""
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
    """Strategy 6: Morphological cleanup - ดีสำหรับภาพที่มี noise เยอะ"""
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # dilate จะทำให้ตัวอักษรชัดขึ้น
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.dilate(gray, kernel, iterations=1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        logger.warning(f"[v6] failed: {e}")
        return None


ALL_PREPROCESS_STRATEGIES = [
    ("v1_clahe", preprocess_v1_clahe),
    ("v2_adaptive", preprocess_v2_adaptive_thresh),
    ("v3_otsu", preprocess_v3_otsu),
    ("v4_sharp", preprocess_v4_sharpened),
    ("v5_inverted", preprocess_v5_inverted),
    ("v6_morph", preprocess_v6_morph),
]


# ==============================
# OCR TEXT CLEANER
# ==============================

def clean_ocr_text(text: str) -> str:
    """ทำความสะอาด text ที่ OCR อ่านมาก่อน match pattern
    
    แก้ปัญหา OCR artifacts เช่น:
    - SSHO1701 → SSH01701 (O แปลงเป็น 0)
    - SRHL00248 → SRH-00248 (L จากเส้น | บนป้ายโดนอ่านเป็นตัวอักษร)
    - SRHG-00267 → SRHG-00267 (prefix 4 ตัวอักษร + dash + ตัวเลข)
    - ABCDE-12345 → ABCDE-12345 (prefix 5 ตัวอักษร)
    """
    text = text.replace(" ", "").replace("\n", "").upper()
    # ลบ _ และ | ออก แต่ยังเก็บ dash (-) ไว้ช่วยแยก prefix
    text = re.sub(r'[_|]', '', text)

    # === 1. SSH-specific cleaning ===
    # หา SSH (รวม OCR ผิด เช่น 5SH, 55H, S5H)
    # สำหรับ SSH: แปลง OCR artifacts เป็นตัวเลข (O→0, I→1) เพราะรหัส SSH ต่อด้วยเลขเสมอ
    digit_map = {"O": "0", "I": "1", "L": "1", "Z": "2", "B": "8", "G": "6", "Q": "0"}
    m_ssh = re.search(r'(SSH|5SH|55H|S5H|5S4)', text)
    if m_ssh:
        prefix = "SSH"
        rest = text[m_ssh.end():]
        rest = re.sub(r'[-]', '', rest)
        cleaned = ""
        for c in rest:
            cleaned += digit_map.get(c, c)
        cleaned = re.sub(r'[^0-9]', '', cleaned)
        if cleaned:
            return prefix + cleaned

    # === 2. มี dash separator → เชื่อ prefix ทั้งหมด (2-5 ตัว) ===
    # เช่น SRHG-00267, AB-1234, EQUIP-001
    m_sep = re.match(r'([A-Z]{2,5})-(\d+)', text)
    if m_sep:
        return m_sep.group(1) + "-" + m_sep.group(2)

    # === 3. ไม่มี dash → prefix 2-3 ตัว + ข้าม artifact ตรง boundary + ใส่ dash + ตัวเลข ===
    # เช่น SRHL00248 → SRH-00248 (skip L, insert dash)
    # เช่น SRHF00258 → SRH-00258 (skip F, insert dash)
    # (prefix 4+ ตัวให้ใช้ dash ช่วยแยกใน step 2)
    text_no_dash = re.sub(r'[-]', '', text)
    m_gen = re.match(r'([A-Z]{2,3})\D*(\d+)', text_no_dash)
    if m_gen:
        return m_gen.group(1) + "-" + m_gen.group(2)

    return text_no_dash


def _is_label_noise(code: str) -> bool:
    """ตรวจว่า code ที่ได้มาเป็นคำที่อยู่บนป้ายแต่ไม่ใช่รหัสครุภัณฑ์"""
    # แยก prefix (ตัวอักษรนำหน้า) ออกมาเช็ค
    prefix_match = re.match(r'([A-Z]+)', code)
    if prefix_match:
        prefix = prefix_match.group(1)
        if prefix in Config.LABEL_NOISE_WORDS:
            return True
    return False


def extract_code_from_text(text: str) -> Optional[Tuple[str, float]]:
    """
    พยายาม extract equipment code จาก text ด้วย pattern matching หลายแบบ
    รองรับทั้ง SSH, SRH, SRHG, และรูปแบบอื่น ๆ
    Returns (code, confidence_score) หรือ None
    """
    text_clean = text.replace(" ", "").upper()

    # Priority 1: Exact SSH (เดิม — confidence สูงสุด)
    m = Config.SSH_PATTERN.search(text_clean)
    if m:
        code = re.sub(r'[^A-Z0-9]', '', m.group(0)).upper()
        return code, 1.0

    # Priority 2: SSH ที่ OCR อ่านผิดเล็กน้อย เช่น 5SH, 55H, SSHO4582
    # ใช้ digit_map แปลง O→0, I→1 ฯลฯ ก่อน match เพื่อไม่ให้ตัวเลขหาย
    digit_map = {"O": "0", "I": "1", "L": "1", "Z": "2", "B": "8", "G": "6", "Q": "0"}
    ssh_prefix = re.search(r'(SSH|5SH|55H|S5H|5S4)', text_clean)
    if ssh_prefix:
        rest = text_clean[ssh_prefix.end():]
        rest = re.sub(r'[-]', '', rest)
        # แปลง OCR artifacts เป็นตัวเลข
        cleaned_digits = ""
        for c in rest:
            cleaned_digits += digit_map.get(c, c)
        cleaned_digits = re.sub(r'[^0-9]', '', cleaned_digits)
        if len(cleaned_digits) >= 3:
            return "SSH" + cleaned_digits, 0.85

    # Priority 2b: Loose match สำหรับ SSH ที่ prefix ก็อ่านผิดด้วย
    loose = re.search(r'[S5]{1,2}[SH5][H0-9]?[\s\-_]*(\d{3,})', text_clean)
    if loose:
        code = "SSH" + loose.group(1)
        return code, 0.75

    # Priority 3: General equipment code — ตัวอักษร 2-5 ตัว + ตัวเลข 2+ หลัก
    # เช่น SRH00248, SRHG00267, ABC1234, EQUIP001
    gen = Config.GENERAL_CODE_PATTERN.search(text_clean)
    if gen:
        code = re.sub(r'[^A-Z0-9\-]', '', gen.group(0)).upper()
        # กรอง false positive: ต้องยาวพอ + ไม่ใช่คำบนป้าย
        if len(code) >= 4 and not _is_label_noise(code):
            return code, 0.9

    # Priority 4: มีตัวเลข 4+ หลักแต่ prefix ไม่ชัด → ลอง SSH เป็น fallback
    nums = re.search(r'(\d{4,})', text_clean)
    if nums and len(text_clean) < 12:
        code = "SSH" + nums.group(1)
        return code, 0.5

    return None


# ==============================
# OCR FUNCTIONS
# ==============================

def run_easyocr(img_gray: np.ndarray, top_crop: bool = True) -> List[Tuple[str, float]]:
    """Run EasyOCR และคืน list ของ (text, confidence)"""
    try:
        h = img_gray.shape[0]
        img = img_gray[int(h * Config.OCR_TOP_CROP_RATIO):, :] if top_crop else img_gray

        results = reader.readtext(
            img,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-",
            paragraph=False,
            detail=1,
        )
        return [(text, conf) for (_, text, conf) in results]
    except Exception as e:
        logger.warning(f"EasyOCR error: {e}")
        return []




def try_all_ocr(processed_img: np.ndarray) -> Optional[Tuple[str, float, str]]:
    """
    ลอง EasyOCR บน processed image
    Returns (code, confidence, engine) หรือ None
    """
    # --- EasyOCR: crop บน ---
    for text, conf in run_easyocr(processed_img, top_crop=True):
        cleaned = clean_ocr_text(text)
        result = extract_code_from_text(cleaned)
        if result:
            code, score = result
            return code, conf * score, "easyocr_cropped"

    # --- EasyOCR: full image (ไม่ crop) ---
    for text, conf in run_easyocr(processed_img, top_crop=False):
        cleaned = clean_ocr_text(text)
        result = extract_code_from_text(cleaned)
        if result:
            code, score = result
            return code, conf * score, "easyocr_full"

    return None


# ==============================
# MAIN DETECTION PIPELINE
# ==============================

def detect_ssh_code(img: np.ndarray) -> Dict:
    start = time.time()
    detected = False

    results = model.predict(
        source=img,
        conf=Config.YOLO_CONF,
        imgsz=Config.YOLO_IMGSZ,
        verbose=False,
        max_det=2,
        device=0,
        half=True
    )

    for r in results:
        if not r.boxes:
            continue
        detected = True

        for box in sorted(r.boxes, key=lambda x: x.conf[0], reverse=True):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ขยาย bounding box เล็กน้อยเผื่อ text ถูก crop ออก
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.shape[1], x2 + pad_x)
            y2 = min(img.shape[0], y2 + pad_y)

            crop = img[y1:y2, x1:x2]
            if not validate_image(crop):
                continue

            logger.info(f"🔍 Trying crop size: {crop.shape} | yolo_conf: {box.conf[0]:.2f}")

            # ลอง preprocess ทุก strategy
            for strategy_name, preprocess_fn in ALL_PREPROCESS_STRATEGIES:
                processed = preprocess_fn(crop)
                if processed is None:
                    continue

                ocr_result = try_all_ocr(processed)
                if ocr_result:
                    code, confidence, engine = ocr_result
                    elapsed = time.time() - start
                    logger.info(f"✅ SUCCESS | code={code} | strategy={strategy_name} | engine={engine} | time={elapsed:.2f}s")
                    return {
                        "status": "success",
                        "message": "Detect และ OCR สำเร็จ",
                        "code": code,
                        "source": f"{engine}+{strategy_name}",
                        "confidence": float(box.conf[0]),
                        "ocr_confidence": round(confidence, 3),
                        "processing_time": f"{elapsed:.2f}s"
                    }

            logger.warning("⚠️ All strategies failed for this box")

    if detected:
        return {
            "status": "detected_no_ocr",
            "message": "Detect เจอป้าย แต่ OCR อ่านรหัสไม่ได้ (ลองแล้วทุก strategy)",
            "processing_time": f"{time.time() - start:.2f}s"
        }

    return {
        "status": "not_detected",
        "message": "YOLO ไม่พบป้าย",
        "processing_time": f"{time.time() - start:.2f}s"
    }


# ==============================
# SAVE JSON
# ==============================

def save_json_result(result: Dict):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    code = result.get("code", "NO_CODE")
    filename = f"{timestamp}_{code}.json"
    path = os.path.join(Config.OUTPUT_JSON_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"📄 JSON saved: {path}")


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

    result = detect_ssh_code(img)
    save_json_result(result)
    return jsonify(result)


# ==============================
# DEBUG ENDPOINT (optional)
# ==============================

@app.route("/debug", methods=["POST"])
def debug_image():
    """
    Endpoint สำหรับ debug: แสดงผลลัพธ์จากทุก strategy
    ไม่ต้องใช้ YOLO ทำ crop เอง ส่งภาพป้ายมาตรง ๆ ได้เลย
    """
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
        found = None
        for text, conf, engine in all_texts:
            cleaned = clean_ocr_text(text)
            r = extract_code_from_text(cleaned)
            if r:
                found = {"code": r[0], "score": r[1], "engine": engine, "raw_text": text}
                break

        debug_results.append({
            "strategy": strategy_name,
            "raw_texts": [(t, round(c, 3)) for t, c, _ in all_texts[:5]],
            "found": found
        })

    return jsonify({"debug": debug_results})


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    logger.info(f"🚀 Starting OCR API on port {Config.PORT}")
    app.run(host="0.0.0.0", port=Config.PORT, debug=False)