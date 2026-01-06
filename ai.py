from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
from pyzbar.pyzbar import decode
import logging
from functools import lru_cache
from typing import Optional, Tuple, Dict
import time
import json
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
    
    # YOLO Settings
    YOLO_CONF = 0.3
    YOLO_IMGSZ = 960
    
    # Image Enhancement
    RESIZE_SCALE = 3
    DENOISE_H = 20
    ADAPTIVE_THRESH_BLOCK = 31
    ADAPTIVE_THRESH_C = 10
    
    # OCR Settings
    OCR_TOP_CROP_RATIO = 0.45  # ตัดส่วนบน 45%
    
    # Pattern
    SSH_PATTERN = re.compile(r'SSH\d{3,}', re.IGNORECASE)

# ==============================
# LOAD MODELS (with error handling)
# ==============================
try:
    model = YOLO(Config.MODEL_PATH)
    logger.info(f"✅ YOLO model loaded from {Config.MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {e}")
    raise

try:
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    logger.info("✅ EasyOCR reader initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize EasyOCR: {e}")
    raise

app = Flask(__name__)

# ==============================
# IMAGE PREPROCESSING
# ==============================
def validate_image(img: np.ndarray) -> bool:
    """ตรวจสอบความถูกต้องของรูปภาพ"""
    if img is None or img.size == 0:
        return False
    if len(img.shape) < 2:
        return False
    if img.shape[0] < 10 or img.shape[1] < 10:  # รูปเล็กเกินไป
        return False
    return True

def enhance_image(img: np.ndarray) -> Optional[np.ndarray]:
    """ปรับปรุงคุณภาพภาพเพื่อ OCR"""
    try:
        if not validate_image(img):
            return None
        
        # Resize for better OCR
        img = cv2.resize(
            img, 
            None, 
            fx=Config.RESIZE_SCALE, 
            fy=Config.RESIZE_SCALE, 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=Config.DENOISE_H)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            Config.ADAPTIVE_THRESH_BLOCK,
            Config.ADAPTIVE_THRESH_C
        )
        
        return thresh
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return None

# ==============================
# QR CODE READER
# ==============================
def read_qr(img: np.ndarray) -> Optional[str]:
    """อ่าน QR Code และหา SSH pattern"""
    try:
        qr_results = decode(img)
        for qr in qr_results:
            try:
                data = qr.data.decode("utf-8").strip()
                m = Config.SSH_PATTERN.search(data)
                if m:
                    logger.info(f"✅ QR Code found: {m.group(0)}")
                    return m.group(0).upper()  # ทำให้เป็นตัวพิมพ์ใหญ่
            except Exception as e:
                logger.warning(f"Error decoding QR data: {e}")
                continue
    except Exception as e:
        logger.warning(f"QR decode error: {e}")
    
    return None

# ==============================
# OCR SSH EXTRACTION
# ==============================
def read_ssh_from_ocr(img: np.ndarray) -> Optional[str]:
    """อ่าน SSH code จาก OCR (ตัดส่วน Health ออก)"""
    try:
        if not validate_image(img):
            return None
        
        h = img.shape[0]
        # ตัดส่วนบนออก (Health info)
        img = img[int(h * Config.OCR_TOP_CROP_RATIO):, :]
        
        if img.size == 0:
            return None
        
        # OCR with specific allowlist
        ocr_results = reader.readtext(
            img,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        )
        
        # หา SSH pattern ที่มี confidence สูงสุด
        best_match = None
        best_conf = 0
        
        for _, text, conf in ocr_results:
            clean_text = text.replace(" ", "").replace("O", "0")  # แก้ O เป็น 0
            m = Config.SSH_PATTERN.search(clean_text)
            if m and conf > best_conf:
                best_match = m.group(0).upper()
                best_conf = conf
        
        if best_match:
            logger.info(f"✅ OCR found: {best_match} (conf: {best_conf:.2f})")
            return best_match
            
    except Exception as e:
        logger.error(f"OCR error: {e}")
    
    return None

# ==============================
# MAIN DETECTION LOGIC
# ==============================
def detect_ssh_code(img: np.ndarray) -> Dict:
    """ตรวจจับ SSH code จากภาพ"""
    start_time = time.time()
    
    try:
        # Run YOLO detection
        results = model.predict(
            source=img,
            conf=Config.YOLO_CONF,
            imgsz=Config.YOLO_IMGSZ,
            verbose=False
        )
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            # เรียงตาม confidence จากมากไปน้อย
            boxes = sorted(r.boxes, key=lambda x: x.conf[0], reverse=True)
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # ตรวจสอบขนาด bounding box
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                
                crop = img[y1:y2, x1:x2]
                
                if not validate_image(crop):
                    continue
                
                logger.info(f"Processing box with confidence: {conf:.2f}")
                
                # 1️⃣ ลอง QR Code ก่อน (เร็วกว่า)
                qr_code = read_qr(crop)
                if qr_code:
                    elapsed = time.time() - start_time
                    return {
                        "status": "success",
                        "code": qr_code,
                        "source": "QR",
                        "confidence": 1.0,
                        "processing_time": f"{elapsed:.2f}s"
                    }
                
                # 2️⃣ ลอง OCR
                processed = enhance_image(crop)
                if processed is not None:
                    ssh_code = read_ssh_from_ocr(processed)
                    
                    if ssh_code:
                        elapsed = time.time() - start_time
                        return {
                            "status": "success",
                            "code": ssh_code,
                            "source": "OCR",
                            "confidence": conf,
                            "processing_time": f"{elapsed:.2f}s"
                        }
        
        # ไม่เจอ SSH code
        elapsed = time.time() - start_time
        return {
            "status": "not_found",
            "message": "ไม่พบรหัสเครื่อง (SSH)",
            "processing_time": f"{elapsed:.2f}s"
        }
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# ==============================
# API ENDPOINTS
# ==============================
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "ocr_ready": reader is not None
    })

@app.route("/ocr", methods=["POST"])
def ocr_image():
    """Main OCR endpoint"""
    try:
        # ตรวจสอบ file upload
        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded"
            }), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({
                "status": "error",
                "message": "Empty filename"
            }), 400
        
        # อ่านภาพ
        try:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return jsonify({
                "status": "error",
                "message": "Invalid image file"
            }), 400
        
        if not validate_image(img):
            return jsonify({
                "status": "error",
                "message": "Invalid or corrupted image"
            }), 400
        
        # ประมวลผล
        result = detect_ssh_code(img)
        
        # กำหนด status code
        if result["status"] == "success":
            return jsonify(result), 200
        elif result["status"] == "not_found":
            return jsonify(result), 404
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

# ==============================
# ERROR HANDLERS
# ==============================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    logger.info(f"🚀 Starting OCR API on port {Config.PORT}")
    app.run(
        host="0.0.0.0",
        port=Config.PORT,
        debug=False,
        threaded=True
    )