from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re

# =============================
# CONFIG
# =============================
MODEL_PATH = r"D:/gog/best.pt"

SSH_PATTERN = re.compile(r'^SSH[0-9A-Z]{3,10}$')

YOLO_CONF_THRESHOLD = 0.7
OCR_CONF_THRESHOLD = 0.5

# =============================
# LOAD MODEL
# =============================
model = YOLO(MODEL_PATH)

# auto gpu fallback
try:
    reader = easyocr.Reader(['en'], gpu=True)
except:
    reader = easyocr.Reader(['en'], gpu=False)

app = Flask(__name__)

# =============================
# IMAGE ENHANCE
# =============================
def enhance_image(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp

# =============================
# NORMALIZE TEXT
# =============================
def normalize_text(text):
    t = text.upper().replace(" ", "").replace("-", "")
    t = (
        t.replace('O', '0')
         .replace('Q', '0')
         .replace('I', '1')
         .replace('L', '1')
         .replace('Z', '2')
    )
    return t

# =============================
# API
# =============================
@app.route("/ocr", methods=["POST"])
def ocr_image():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # ---------------- YOLO DETECT ----------------
    results = model.predict(
        source=img,
        conf=YOLO_CONF_THRESHOLD,
        imgsz=800,
        verbose=False
    )

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf = float(box.conf[0])

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            processed = enhance_image(crop)

            # ---------------- OCR ----------------
            ocr_result = reader.readtext(
                processed,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            texts = []
            for _, text, ocr_conf in ocr_result:
                if ocr_conf < OCR_CONF_THRESHOLD:
                    continue

                norm = normalize_text(text)

                texts.append({
                    "raw": text,
                    "normalized": norm,
                    "confidence": round(float(ocr_conf), 3),
                    "is_ssh": bool(SSH_PATTERN.match(norm))
                })

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "yolo_confidence": round(yolo_conf, 3),
                "ocr_results": texts
            })

    return jsonify({
        "status": "success",
        "count": len(detections),
        "detections": detections
    })

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
