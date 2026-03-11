from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import re
from pyzbar.pyzbar import decode
import logging
import time
import json
import os
import pytesseract
from typing import Optional, Dict

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================
# CONFIG
# ==============================
class Config:
    MODEL_PATH = "D:/gog/AI_ocr/best_new.pt"
    PORT = 8000

    OUTPUT_JSON_DIR = "json_results"

    YOLO_CONF = 0.3
    YOLO_IMGSZ = 960

    OCR_TOP_CROP_RATIO = 0.35
    SSH_PATTERN = re.compile(r"SSH\d{3,}", re.IGNORECASE)

    TESS_CONFIG = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

# ==============================
# INIT
# ==============================
os.makedirs(Config.OUTPUT_JSON_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r"D:/Tesseract-OCR/tesseract.exe"
model = YOLO(Config.MODEL_PATH)

app = Flask(__name__)

# ==============================
# UTILS
# ==============================
def validate_image(img: np.ndarray) -> bool:
    return img is not None and img.size > 0 and img.shape[0] > 10 and img.shape[1] > 10


def preprocess_for_tesseract(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU → Tesseract ชอบสุด
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # เพิ่มความหนาตัวอักษร
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.dilate(th, kernel, iterations=1)

    return th


def read_qr(img: np.ndarray) -> Optional[str]:
    for qr in decode(img):
        try:
            txt = qr.data.decode("utf-8")
            m = Config.SSH_PATTERN.search(txt)
            if m:
                return m.group(0).upper()
        except:
            pass
    return None


def read_ssh_from_tesseract(img: np.ndarray) -> Optional[str]:
    h = img.shape[0]
    img = img[int(h * Config.OCR_TOP_CROP_RATIO):, :]

    img = preprocess_for_tesseract(img)

    data = pytesseract.image_to_data(
        img,
        config=Config.TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    texts, confs = [], []
    for txt, conf in zip(data["text"], data["conf"]):
        if conf != "-1":
            txt = txt.replace(" ", "").replace("O", "0").upper()
            if txt:
                texts.append(txt)
                confs.append(float(conf))

    merged = "".join(texts)
    m = Config.SSH_PATTERN.search(merged)
    if not m:
        return None

    return m.group(0)


# ==============================
# MAIN LOGIC
# ==============================
def detect_ssh_code(img: np.ndarray) -> Dict:
    start = time.time()

    preds = model.predict(
        source=img,
        conf=Config.YOLO_CONF,
        imgsz=Config.YOLO_IMGSZ,
        verbose=False
    )

    for r in preds:
        if not r.boxes:
            continue

        for box in sorted(r.boxes, key=lambda b: b.conf[0], reverse=True):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            if not validate_image(crop):
                continue

            # 1️⃣ QR
            qr = read_qr(crop)
            if qr:
                return {
                    "status": "success",
                    "code": qr,
                    "source": "QR",
                    "confidence": 1.0,
                    "processing_time": f"{time.time() - start:.2f}s"
                }

            # 2️⃣ Tesseract
            ocr = read_ssh_from_tesseract(crop)
            if ocr:
                return {
                    "status": "success",
                    "code": ocr,
                    "source": "TESSERACT",
                    "confidence": round(float(box.conf[0]), 3),
                    "processing_time": f"{time.time() - start:.2f}s"
                }

    return {
        "status": "not_found",
        "message": "ไม่พบรหัสเครื่อง (SSH)",
        "processing_time": f"{time.time() - start:.2f}s"
    }

# ==============================
# SAVE JSON
# ==============================
def save_json_result(result: Dict):
    ts = time.strftime("%Y%m%d_%H%M%S")
    code = result.get("code", "NO_CODE")
    path = os.path.join(Config.OUTPUT_JSON_DIR, f"{ts}_{code}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

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
# RUN
# ==============================
if __name__ == "__main__":
    logger.info(f"🚀 OCR API (Tesseract-only) running on port {Config.PORT}")
    app.run(host="0.0.0.0", port=Config.PORT, debug=False)
