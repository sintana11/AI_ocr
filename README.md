# 🏥 Medical Equipment OCR API

ระบบ **AI OCR** สำหรับอ่านรหัสครุภัณฑ์ (Equipment ID) จากภาพถ่ายป้ายอุปกรณ์การแพทย์
ใช้ **YOLOv11** ตรวจจับตำแหน่งป้าย + **EasyOCR** อ่านข้อความ พร้อม preprocessing pipeline หลายรูปแบบเพื่อเพิ่มความแม่นยำ

---

## 🛠️ Tech Stack — ภาษาและ Framework ที่ใช้

### ภาษาโปรแกรม

| ภาษา | เวอร์ชัน | บทบาท |
|---|---|---|
| **Python** | 3.10+ | ภาษาหลักทั้งหมดของโปรเจกต์ — ทั้ง API server, AI model inference, image processing, และ OCR |

### Frameworks & Libraries

| Framework / Library | เวอร์ชัน | ประเภท | หน้าที่ |
|---|---|---|---|
| **Flask** | ≥ 3.0 | Web Framework | สร้าง REST API server รับภาพจาก client แล้วส่งผลลัพธ์กลับเป็น JSON |
| **Ultralytics (YOLOv11)** | ≥ 8.0 | Object Detection | ตรวจจับตำแหน่งป้ายรหัสครุภัณฑ์ในภาพ ใช้ YOLO model ที่ train เอง |
| **EasyOCR** | ≥ 1.7 | OCR Engine | อ่านตัวอักษร/ตัวเลขจากภาพ crop ที่ได้จาก YOLO — รองรับภาษาอังกฤษ |
| **OpenCV** (`opencv-python-headless`) | ≥ 4.8 | Image Processing | ปรับแต่งภาพก่อน OCR เช่น resize, grayscale, threshold, blur, sharpen, morphology |
| **NumPy** | ≥ 1.24 | Numerical Computing | จัดการ array ของภาพ, คำนวณ matrix operations สำหรับ image processing |

### Infrastructure

| เครื่องมือ | หน้าที่ |
|---|---|
| **Git** | Version control สำหรับ source code |

### Python Standard Libraries ที่ใช้

| Library | หน้าที่ |
|---|---|
| `re` | Regular Expression — ใช้ clean text, กรอง noise, ตรวจ pattern ตัวอักษร/ตัวเลข |
| `logging` | บันทึก log ทุก step ของ pipeline เพื่อ debug และ monitoring |
| `collections` (`Counter`, `defaultdict`) | นับ vote ของ OCR candidates, จัดกลุ่มผลลัพธ์ |
| `typing` | Type hints เพื่อให้โค้ดอ่านง่ายและตรวจสอบได้ |
| `time` | วัดเวลาประมวลผลแต่ละ request |
| `gc` | Garbage Collection — บังคับคืน memory หลังประมวลผลภาพ |
| `os` | อ่าน environment variables สำหรับ configuration |
| `json` | จัดการ JSON data |

---

## ✨ Features

| Feature | รายละเอียด |
|---|---|
| **YOLO Object Detection** | ตรวจจับตำแหน่งป้ายรหัสครุภัณฑ์ในภาพ (YOLOv11) |
| **Multi-Strategy Preprocessing** | 8 วิธีปรับภาพ (CLAHE, Adaptive Threshold, Otsu, Sharpened, Inverted, Morphology, Bilateral, High-Res) |
| **Voting-based OCR** | ใช้ระบบ vote จากหลาย strategy เพื่อเลือกผลลัพธ์ที่ดีที่สุด |
| **OCR Correction** | แก้ไข misread อัตโนมัติ เช่น `8→S`, `O→0`, `I→1` ตาม context ตำแหน่ง |
| **Noise Filtering** | กรองคำบนป้ายที่ไม่ใช่รหัส เช่น "HEALTH", "CALIBRATION", "MODEL" |
| **Adaptive Resize** | ปรับขนาด crop อัตโนมัติ — crop เล็กจะถูกขยาย, crop ใหญ่ไม่ขยาย |
| **Memory Optimized** | จำกัดขนาดภาพ + `gc.collect()` คืน RAM หลังประมวลผล |

---

## 🏗️ Architecture

### System Overview

```
Client (Mobile/Web App)
         │
         │  POST /ocr  (multipart/form-data — ส่งไฟล์ภาพ)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask REST API Server                     │
│                     (port 8000)                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  OCR Pipeline                         │  │
│  │                                                       │  │
│  │  1. Image Validation & Resize                         │  │
│  │       ↓                                               │  │
│  │  2. YOLO v11 Detection  ──→  ตรวจจับป้ายรหัส          │  │
│  │       ↓                                               │  │
│  │  3. Crop + Padding (20%)                              │  │
│  │       ↓                                               │  │
│  │  4. Multi-Strategy Preprocessing (8 วิธี)              │  │
│  │       ↓                                               │  │
│  │  5. EasyOCR + Rotation (0°, ±2°)                      │  │
│  │       ↓                                               │  │
│  │  6. Noise Filtering & Text Extraction                 │  │
│  │       ↓                                               │  │
│  │  7. Voting → OCR Correction → Final Result            │  │
│  └───────────────────────────────────────────────────────┘  │
│         │                                                    │
│         ▼                                                    │
│    JSON Response  ──→  { code, confidence, status }          │
└─────────────────────────────────────────────────────────────┘
```

### Communication Flow

1. **Client** ส่ง HTTP `POST` request พร้อมไฟล์ภาพ (JPEG/PNG) ผ่าน `multipart/form-data`
2. **Flask** รับ request ที่ endpoint `/ocr` แล้วส่งเข้า pipeline
3. **Pipeline** ประมวลผลภาพผ่าน 7 ขั้นตอน แล้วส่ง JSON response กลับ
4. **Response format** เป็น JSON มาตรฐาน — พร้อมใช้กับ frontend ใดก็ได้

---

## 🔄 How It Works — รูปแบบการทำงานอย่างละเอียด

### ขั้นตอนที่ 1: รับภาพและ Validation

```
ภาพจาก client  →  cv2.imdecode()  →  validate_image()  →  limit_image_size()
```

- Flask รับไฟล์ภาพจาก `request.files["file"]`
- ใช้ OpenCV decode ภาพจาก binary เป็น numpy array (BGR format)
- ตรวจสอบว่าภาพถูกต้อง: ไม่ใช่ null, มีขนาดอย่างน้อย 10×10 pixels
- **จำกัดขนาดภาพไม่เกิน 1280px** (ด้านยาวสุด) เพื่อลดการใช้ RAM ตลอด pipeline

### ขั้นตอนที่ 2: YOLO Object Detection

```
ภาพ  →  YOLO v11 predict()  →  Bounding Box ของป้ายรหัส
```

- ใช้ **YOLOv11 Nano** (model ที่ train มาเฉพาะสำหรับตรวจจับป้ายรหัสครุภัณฑ์)
- Parameters: `conf=0.25`, `imgsz=640`, `max_det=1` (ตรวจจับแค่ 1 ป้ายต่อภาพ)
- รันบน **CPU** (ไม่ต้องใช้ GPU)
- ผลลัพธ์: พิกัด bounding box `(x1, y1, x2, y2)` และค่า confidence

### ขั้นตอนที่ 3: Crop ภาพ + Padding

```
Bounding Box  →  เพิ่ม padding 20%  →  crop จากภาพต้นฉบับ
```

- ตัดภาพเฉพาะบริเวณป้ายที่ YOLO ตรวจเจอ
- **เพิ่ม padding 20%** รอบ ๆ เพื่อให้ OCR มี context มากขึ้น (ตัวอักษรริมขอบไม่ถูกตัด)
- คำนวณ **adaptive scale** — ถ้า crop เล็กกว่า 300px จะขยายอัตโนมัติ (สูงสุด 4x)

### ขั้นตอนที่ 4: Multi-Strategy Preprocessing

```
crop  →  [v1_clahe, v3_otsu, v2_adaptive, ...]  →  ภาพปรับแต่งหลายแบบ
```

ใช้ **Fast Strategy** (v1, v3, v2) ก่อน — ถ้าได้ confidence ≥ 0.35 จะหยุดทันที ไม่ต้องลองทุกวิธี

| Pass | กลยุทธ์ | เมื่อไหร่ |
|---|---|---|
| **Pass 1** | อ่านภาพเต็ม (ไม่ crop top) ด้วย Fast Strategies | ลองทุกครั้ง |
| **Pass 2** | ตัดครึ่งบนออก (top crop 45%) ดูแค่ส่วนล่างที่มีรหัส | เมื่อ Pass 1 ไม่เจอ หรือ confidence < 0.25 |

แต่ละ strategy ใช้ OpenCV ปรับภาพต่างกัน:
- **Grayscale** → แปลงภาพสีเป็นขาวดำ
- **CLAHE** → ปรับ contrast แบบ adaptive (ไม่ทำให้ส่วนสว่างจ้าเกินไป)
- **Threshold** → แปลงเป็นภาพ binary (ขาว/ดำ) เพื่อแยกข้อความจากพื้นหลัง
- **Blur/Sharpen** → ลด noise หรือเพิ่มความคมชัดตัวอักษร
- **Morphology** → เชื่อมส่วนที่ขาดของตัวอักษร

### ขั้นตอนที่ 5: EasyOCR + Rotation

```
ภาพปรับแต่ง  →  EasyOCR (0°, +2°, -2°)  →  [(text, confidence), ...]
```

- ใช้ **EasyOCR** อ่านตัวอักษร/ตัวเลขจากภาพ
- **Allowlist**: `A-Z`, `0-9`, `-` (ไม่อ่านตัวอักษรพิเศษ)
- ลองหมุนภาพ ±2° เพื่อรับมือกับภาพเอียงเล็กน้อย
- แต่ละ strategy × แต่ละมุมหมุน = candidate ตัวเลือกหนึ่ง

### ขั้นตอนที่ 6: Noise Filtering & Text Extraction

```
OCR results  →  clean_ocr_text()  →  _is_label_noise()  →  extract_text_no_pattern()
```

- **Clean**: ลบ whitespace, newline, อักขระพิเศษ
- **Noise filter**: กรองคำบนป้ายที่ไม่ใช่รหัส เช่น `HEALTH`, `BIOMEDICAL`, `CALIBRATION`, `SERIAL`
- **Validation**: ต้องมีทั้งตัวอักษร (A-Z) และตัวเลข (0-9) อย่างน้อย 3 ตัว, ยาวอย่างน้อย 4 ตัวอักษร
- ผลลัพธ์: `(cleaned_text, score)` — score คำนวณจากความยาวข้อความ

### ขั้นตอนที่ 7: Voting + OCR Correction

```
candidates  →  pick_best_candidate()  →  correct_ocr_format()  →  Final Result
```

#### 7a. Voting System
- รวม candidates จากทุก strategy + ทุกมุมหมุน
- นับจำนวน **vote** ของแต่ละ text ที่ซ้ำกัน
- เรียงลำดับตาม: จำนวน votes → confidence สูงสุด
- เลือก text ที่ได้ votes มากที่สุด

#### 7b. OCR Correction (`correct_ocr_format`)
แก้ตัวอักษรที่ OCR อ่านผิดบ่อย โดยวิเคราะห์ **context ตำแหน่ง**:

| ตำแหน่ง | หลักการ | ตัวอย่าง |
|---|---|---|
| **Prefix** (3 ตัวแรก) | ควรเป็นตัวอักษร → แก้เลขเป็นอักษร | `8` → `S`, `5` → `S`, `0` → `O` |
| **Suffix** (ที่เหลือ) | ควรเป็นตัวเลข → แก้อักษรเป็นเลข | `O` → `0`, `I` → `1`, `S` → `5` |

ตัวอย่าง: `89H04622` → `SSH04622` (แก้ `89` เป็น `SS` เพราะอยู่ในตำแหน่ง prefix)

#### 7c. Final Response
- ตรวจสอบ OCR confidence ≥ 0.20 ก่อนส่งผลลัพธ์
- ส่ง JSON response พร้อม: `code`, `confidence`, `ocr_confidence`, `source`, `processing_time`
- ปล่อย memory ด้วย `del img` + `gc.collect()`

---

## 📁 Project Structure

```
AI_ocr/
├── ai_no_pattern.py       # 🎯 Main application — Flask API + OCR pipeline (757 lines)
├── model/                 # 🧠 YOLO model weights (train มาเฉพาะป้ายครุภัณฑ์)
│   └── V11n/              #    └── YOLOv11 Nano — ใช้ใน production
│       └── weights/
│           └── best.pt    #        model weights ที่ train แล้ว
├── requirements.txt       # 📦 Python dependencies (5 packages)
└── .gitignore             # 🚫 Git exclusions
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 หรือสูงกว่า
- **YOLO model weights** ใน `model/V11n/weights/best.pt` (model ที่ train มาแล้ว)
- **RAM** อย่างน้อย 512MB (แนะนำ 2GB สำหรับ production)

### Run Locally

```bash
# สร้าง virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน server
python ai_no_pattern.py
```

Server จะรันที่ `http://localhost:8000`

---

## 📡 API Endpoints

### `POST /ocr` — อ่านรหัสครุภัณฑ์

ส่งภาพถ่ายป้ายอุปกรณ์ เพื่ออ่านรหัสครุภัณฑ์

**Request:**
```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@photo.jpg"
```

**Response (สำเร็จ):**
```json
{
  "status": "success",
  "message": "Detect และ OCR สำเร็จ",
  "code": "SSH04622",
  "source": "easyocr_full+v1_clahe",
  "confidence": 0.92,
  "ocr_confidence": 0.847,
  "total_candidates": 5,
  "processing_time": "1.23s"
}
```

| Field | Type | รายละเอียด |
|---|---|---|
| `status` | string | `"success"`, `"detected_no_ocr"`, หรือ `"not_detected"` |
| `code` | string | รหัสครุภัณฑ์ที่อ่านได้ (เฉพาะเมื่อ success) |
| `confidence` | float | ค่า confidence ของ YOLO detection (0.0–1.0) |
| `ocr_confidence` | float | ค่า confidence ของ OCR (0.0–1.0) |
| `source` | string | engine + strategy ที่ให้ผลลัพธ์ดีที่สุด |
| `total_candidates` | int | จำนวน candidates ทั้งหมดที่ได้จากทุก strategy |
| `processing_time` | string | เวลาประมวลผลรวม |

**Response (ตรวจจับได้แต่อ่านไม่ได้):**
```json
{
  "status": "detected_no_ocr",
  "message": "Detect เจอป้าย แต่ OCR อ่านข้อความไม่ได้เลย",
  "processing_time": "2.10s"
}
```

**Response (ไม่พบป้าย):**
```json
{
  "status": "not_detected",
  "message": "YOLO ไม่พบป้าย",
  "processing_time": "0.45s"
}
```

---

### `POST /debug` — Debug OCR ทุก Strategy

ทดสอบภาพกับทุก preprocessing strategy เพื่อเปรียบเทียบผลลัพธ์ — ใช้สำหรับ debug และ tune performance

**Request:**
```bash
curl -X POST http://localhost:8000/debug \
  -F "file=@photo.jpg"
```

**Response:**
```json
{
  "debug": [
    {
      "strategy": "v1_clahe",
      "raw_texts": [["SSH04622", 0.912], ["HEALTH", 0.845]],
      "found": [{"text": "SSH04622", "score": 1.0, "engine": "easyocr", "raw_text": "SSH04622"}]
    },
    {
      "strategy": "v2_adaptive",
      "raw_texts": [],
      "found": []
    }
  ]
}
```

---

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

## ⚙️ Configuration

ตั้งค่าผ่าน **Environment Variables** หรือแก้ไขใน class `Config`:

| Variable | Default | รายละเอียด |
|---|---|---|
| `MODEL_PATH` | `model/V11n/weights/best.pt` | Path ไปยังไฟล์ YOLO model |
| `PORT` | `8000` | Port ที่ API รัน |

### Internal Parameters (class `Config`)

| Parameter | Value | รายละเอียด |
|---|---|---|
| `YOLO_CONF` | `0.1` | YOLO confidence threshold |
| `YOLO_IMGSZ` | `640` | ขนาดภาพสำหรับ YOLO inference |
| `MAX_INPUT_SIZE` | `1280` | จำกัดขนาดภาพ input (ลด RAM) |
| `MIN_OCR_DIM` | `300` | ขนาดขั้นต่ำของ crop สำหรับ OCR |
| `OCR_TOP_CROP_RATIO` | `0.45` | สัดส่วนตัดส่วนบนออก (top crop) |
| `DENOISE_H` | `20` | ค่า denoising |

---

## 🔬 Preprocessing Strategies

ระบบมี 8 วิธีปรับภาพก่อนส่งให้ OCR:

| # | Strategy | เทคนิค | จุดเด่น |
|---|---|---|---|
| 1 | `v1_clahe` | CLAHE contrast | เร็ว, ปรับ contrast ดี |
| 2 | `v2_adaptive` | Adaptive Threshold + Gaussian Blur | ดีกับแสงไม่สม่ำเสมอ |
| 3 | `v3_otsu` | Otsu's Thresholding | แยกพื้นหลัง-ข้อความอัตโนมัติ |
| 4 | `v4_sharp` | Sharpening + CLAHE | เพิ่มความคมชัดตัวอักษร |
| 5 | `v5_inverted` | Invert + CLAHE | สำหรับป้ายสีเข้ม |
| 6 | `v6_morph` | Morphology + Otsu | เชื่อมตัวอักษรที่ขาด |
| 7 | `v7_bilateral` | Bilateral Filter + CLAHE (3x resize) | ลด noise แต่คง edge |
| 8 | `v8_highres` | High-Res 4x + Otsu | สำหรับ crop ขนาดเล็กมาก |

> **Note:** ใน production จะใช้เฉพาะ `v1_clahe`, `v3_otsu`, `v2_adaptive` ก่อน (Fast Strategies) ถ้าได้ confidence ≥ 0.35 จะหยุดทันทีเพื่อประหยัดเวลา

---

## 📦 Dependencies

| Package | Version | ประเภท | หน้าที่ |
|---|---|---|---|
| `flask` | ≥ 3.0 | Web Framework | สร้าง REST API รับ/ส่ง request |
| `ultralytics` | ≥ 8.0 | AI / Object Detection | YOLO model inference ตรวจจับป้าย |
| `easyocr` | ≥ 1.7 | AI / OCR | อ่านตัวอักษรจากภาพ |
| `opencv-python-headless` | ≥ 4.8 | Image Processing | ปรับแต่งภาพ (resize, threshold, blur, etc.) |
| `numpy` | ≥ 1.24 | Numerical | จัดการ array/matrix ของภาพ |

---

## 📝 License

Internal use only.
