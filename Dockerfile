FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PADDLE_PDX_CACHE_HOME=/tmp/paddlex-cache \
    SEAL_DETECTOR_MODEL=models/seal-det-v1.0.0/best.onnx \
    RECOGNITION_MODEL_DIR=models/seal-rec-v1.0.4 \
    MODEL_VERSION=seal-det-v1.0.0-rec-v1.0.4 \
    PORT=7860

WORKDIR /app

# Install system dependencies for OpenCV and Paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libgl1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as non-root user (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application, models, benchmark and baseline data
COPY --chown=user:user app ./app
COPY --chown=user:user models ./models
COPY --chown=user:user benchmark ./benchmark
COPY --chown=user:user baselines ./baselines
COPY --chown=user:user scripts ./scripts

# Ensure cache directory is writable by user 1000
RUN mkdir -p /tmp/paddlex-cache && chown -R user:user /tmp/paddlex-cache /app

USER user

# Pre-fetch base PaddleOCR models into cache
RUN python -m scripts.prefetch_models

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
