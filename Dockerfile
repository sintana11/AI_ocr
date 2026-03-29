# ── Stage: Runtime ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + model
COPY ai_no_pattern.py .
COPY model/V11n/weights/best.pt model/V11n/weights/best.pt

EXPOSE 8000

CMD ["python", "ai_no_pattern.py"]
