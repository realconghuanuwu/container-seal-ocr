FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/service \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex-cache

WORKDIR /service
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 5 -r requirements.txt
COPY app ./app
COPY scripts ./scripts
RUN python -m scripts.prefetch_models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
