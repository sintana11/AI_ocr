from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import os
import uuid

# ---------- CONFIG ----------
MODEL_PATH = "D:/gog/best.pt"
SAVE_DIR = "/content/drive/MyDrive/ocr_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- LOAD MODEL ----------
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)

app = Flask(__name__)

# ---------- IMAGE ENHANCE ----------
def enhance_image(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp

# ---------- API ----------
@app.route("/ocr", methods=["POST"])
def ocr_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model.predict(
        source=img,
        conf=0.2,
        imgsz=800
    )

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            processed = enhance_image(crop)

            crop_name = f"{uuid.uuid4().hex}.png"
            crop_path = os.path.join(SAVE_DIR, crop_name)
            cv2.imwrite(crop_path, processed)

            ocr_result = reader.readtext(processed)

            texts = []
            for _, text, ocr_conf in ocr_result:
                texts.append({
                    "text": text,
                    "confidence": float(ocr_conf)
                })

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "yolo_confidence": conf,
                "ocr_texts": texts,
                "crop_path": crop_path
            })

    return jsonify({
        "status": "success",
        "detections": detections
    })

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
