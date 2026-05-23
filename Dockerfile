FROM python:3.10-slim

# ติดตั้ง system dependencies สำหรับ OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง Python dependencies (แยก layer เพื่อ cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอก model
COPY model/ /app/model/

# คัดลอก application code
COPY ai_no_pattern.py .

# ตั้งค่า environment variables
ENV MODEL_PATH=/app/model/V11n/weights/best.pt
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "ai_no_pattern.py"]
