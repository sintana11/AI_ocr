from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
from pyzbar.pyzbar import decode
import logging
from typing import Optional, Dict
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

# ==============================
# CONFIG
# ==============================
class Config:
    MODEL_PATH = "D:/gog/AI_ocr/best.pt"
    PORT = 8000

    OUTPUT_JSON_DIR = "json_results"   # 🔹 โฟลเดอร์เก็บไฟล์ json

    YOLO_CONF = 0.3
    YOLO_IMGSZ = 960

    RESIZE_SCALE = 3
    DENOISE_H = 20
    ADAPTIVE_THRESH_BLOCK = 31
    ADAPTIVE_THRESH_C = 10

    OCR_TOP_CROP_RATIO = 0.45
    SSH_PATTERN = re.compile(r'SSH\d{3,}', re.IGNORECASE)

# ==============================
# PREPARE OUTPUT DIR
# ==============================
os.makedirs(Config.OUTPUT_JSON_DIR, exist_ok=True)

# ==============================
# LOAD MODELS
# ==============================
model = YOLO(Config.MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

app = Flask(__name__)

# ==============================
# IMAGE UTILS
# ==============================
def validate_image(img: np.ndarray) -> bool:
    return (
        img is not None and
        img.size > 0 and
        len(img.shape) >= 2 and
        img.shape[0] >= 10 and
        img.shape[1] >= 10
    )

def enhance_image(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        img = cv2.resize(img, None, fx=Config.RESIZE_SCALE, fy=Config.RESIZE_SCALE, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=Config.DENOISE_H)
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            Config.ADAPTIVE_THRESH_BLOCK,
            Config.ADAPTIVE_THRESH_C
        )
    except:
        return None

# ==============================
# QR + OCR
# ==============================
def read_qr(img: np.ndarray) -> Optional[str]:
    for qr in decode(img):
        try:
            data = qr.data.decode("utf-8")
            m = Config.SSH_PATTERN.search(data)
            if m:
                return m.group(0).upper()
        except:
            pass
    return None

def read_ssh_from_ocr(img: np.ndarray) -> Optional[str]:
    h = img.shape[0]
    img = img[int(h * Config.OCR_TOP_CROP_RATIO):, :]

    results = reader.readtext(
        img,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )

    best, best_conf = None, 0
    for _, text, conf in results:
        text = text.replace(" ", "").replace("O", "0")
        m = Config.SSH_PATTERN.search(text)
        if m and conf > best_conf:
            best = m.group(0).upper()
            best_conf = conf

    return best

# ==============================
# MAIN LOGIC
# ==============================
def detect_ssh_code(img: np.ndarray) -> Dict:
    start = time.time()

    results = model.predict(
        source=img,
        conf=Config.YOLO_CONF,
        imgsz=Config.YOLO_IMGSZ,
        verbose=False
    )

    for r in results:
        if not r.boxes:
            continue

        for box in sorted(r.boxes, key=lambda x: x.conf[0], reverse=True):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            if not validate_image(crop):
                continue

            qr = read_qr(crop)
            if qr:
                return {
                    "status": "success",
                    "code": qr,
                    "source": "QR",
                    "confidence": 1.0,
                    "processing_time": f"{time.time() - start:.2f}s"
                }

            processed = enhance_image(crop)
            if processed is not None:
                ocr = read_ssh_from_ocr(processed)
                if ocr:
                    return {
                        "status": "success",
                        "code": ocr,
                        "source": "OCR",
                        "confidence": float(box.conf[0]),
                        "processing_time": f"{time.time() - start:.2f}s"
                    }

    return {
        "status": "not_found",
        "message": "ไม่พบรหัสเครื่อง (SSH)",
        "processing_time": f"{time.time() - start:.2f}s"
    }

# ==============================
# SAVE RESULT TO JSON FILE
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

    # 🔹 บันทึกไฟล์ JSON เพิ่ม
    save_json_result(result)

    # 🔹 ตอบ API เป็น JSON เหมือนเดิม
    return jsonify(result)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    logger.info(f"🚀 Starting OCR API on port {Config.PORT}")
    app.run(host="0.0.0.0", port=Config.PORT, debug=False)
