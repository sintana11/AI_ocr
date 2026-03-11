# =====================================================
# 🔥 ENV (ต้องอยู่บนสุดก่อน import ทุกอย่าง)
# =====================================================
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_cuda"] = "0"

# =====================================================
# IMPORTS
# =====================================================
from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
from pyzbar.pyzbar import decode
import logging
import time
import json
import pytesseract

# =====================================================
# TESSERACT PATH
# =====================================================
pytesseract.pytesseract.tesseract_cmd = r"D:/Tesseract-OCR/tesseract.exe"

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIG
# =====================================================
class Config:
    MODEL_YOLO_V8  = "D:/gog/AI_ocr/model/best_v8.pt"
    MODEL_YOLO_V11 = "D:/gog/AI_ocr/best_new.pt"

    PORT = 8000
    OUTPUT_JSON_DIR = "json_results"

    YOLO_CONF = 0.3
    YOLO_IMGSZ = 960

    RESIZE_SCALE = 3
    DENOISE_H = 15

    SSH_PATTERN = re.compile(r"SSH\d{3,}", re.IGNORECASE)

    TESS_CONFIG = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

# =====================================================
# INIT
# =====================================================
os.makedirs(Config.OUTPUT_JSON_DIR, exist_ok=True)

model_v8  = YOLO(Config.MODEL_YOLO_V8)
model_v11 = YOLO(Config.MODEL_YOLO_V11)

easy_reader = easyocr.Reader(["en"], verbose=False)

app = Flask(__name__)

# =====================================================
# UTILS
# =====================================================
def validate_image(img):
    return (
        img is not None and
        img.size > 0 and
        img.shape[0] > 20 and
        img.shape[1] > 20
    )

def preprocess_common(img):
    """
    preprocess กลาง ใช้ร่วมกันทั้ง EasyOCR + Tesseract
    """
    img = cv2.resize(
        img, None,
        fx=Config.RESIZE_SCALE,
        fy=Config.RESIZE_SCALE,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=Config.DENOISE_H)
    return gray

def preprocess_for_tesseract(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return th

def read_qr(img):
    for qr in decode(img):
        try:
            txt = qr.data.decode("utf-8")
            m = Config.SSH_PATTERN.search(txt)
            if m:
                return m.group(0).upper()
        except:
            pass
    return None

# =====================================================
# OCR ENGINES
# =====================================================
def easyocr_read(gray):
    results = easy_reader.readtext(
        gray,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    best, conf = None, 0.0
    for _, text, c in results:
        text = text.replace(" ", "").replace("O", "0")
        m = Config.SSH_PATTERN.search(text)
        if m and c > conf:
            best, conf = m.group(0).upper(), float(c)
    return best, round(conf, 3)

def tesseract_read(gray):
    th = preprocess_for_tesseract(gray)
    txt = pytesseract.image_to_string(th, config=Config.TESS_CONFIG)
    txt = txt.replace(" ", "").replace("\n", "").replace("O", "0")

    m = Config.SSH_PATTERN.search(txt)
    if not m:
        return None, 0.0

    # heuristic confidence ให้ใกล้ EasyOCR
    length = len(m.group(0))
    conf = 0.75 if length < 7 else 0.9
    return m.group(0).upper(), conf

# =====================================================
# YOLO + OCR PIPELINE
# =====================================================
def run_model(model, img):
    result = {
        "detected": False,
        "yolo_confidence": 0.0,
        "final_text": None,
        "easyocr": {"text": None, "confidence": 0.0},
        "tesseract": {"text": None, "confidence": 0.0},
    }

    preds = model.predict(
        source=img,
        conf=Config.YOLO_CONF,
        imgsz=Config.YOLO_IMGSZ,
        max_det=1,
        verbose=False
    )

    for r in preds:
        if not r.boxes:
            continue

        box = r.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        if not validate_image(crop):
            continue

        result["detected"] = True
        result["yolo_confidence"] = float(box.conf[0])

        gray = preprocess_common(crop)

        e_text, e_conf = easyocr_read(gray)
        t_text, t_conf = tesseract_read(gray)

        result["easyocr"] = {"text": e_text, "confidence": e_conf}
        result["tesseract"] = {"text": t_text, "confidence": t_conf}

        votes = [(e_text, e_conf), (t_text, t_conf)]
        votes = [v for v in votes if v[0]]

        if votes:
            result["final_text"] = max(votes, key=lambda x: x[1])[0]

        break

    return result

# =====================================================
# MAIN LOGIC
# =====================================================
def detect_ssh_code(img):
    start = time.time()
    qr = read_qr(img)

    y8  = run_model(model_v8, img)
    y11 = run_model(model_v11, img)

    return {
        "status": "success" if (qr or y8["final_text"] or y11["final_text"]) else "not_found",
        "processing_time": f"{time.time() - start:.2f}s",
        "results": {
            "qr": qr,
            "yolo_v8": y8,
            "yolo_v11": y11
        }
    }

# =====================================================
# SAVE JSON
# =====================================================
def save_json_result(result):
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(Config.OUTPUT_JSON_DIR, f"{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# =====================================================
# API
# =====================================================
@app.route("/ocr", methods=["POST"])
def ocr_image():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "no file"}), 400

    img = cv2.imdecode(
        np.frombuffer(request.files["file"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if not validate_image(img):
        return jsonify({"status": "error", "message": "invalid image"}), 400

    result = detect_ssh_code(img)
    save_json_result(result)

    return jsonify(result)

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    logger.info(f"🚀 OCR API running on port {Config.PORT}")
    app.run(host="0.0.0.0", port=Config.PORT, debug=False)
