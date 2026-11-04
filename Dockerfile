# =============================================================================
# Stage 1: Build React Frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies using clean install
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy frontend source files and build production static bundle
COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Production Python Backend & Single-Container Distribution
# =============================================================================
FROM python:3.11-slim

# Prevent Python from writing bytecode and enable unbuffered console logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEEPDETECT_MODEL=mobilenet_xgboost \
    DEEPDETECT_DB=/app/data/deepdetect.db \
    PORT=8000

WORKDIR /app

# Install system dependencies needed for OpenCV, FFmpeg video decoding, and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using PyTorch CPU index (keeps image lightweight ~1.2 GB without 2.5+ GB CUDA wheels)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy offline InsightFace face detection models and trained detector weights
COPY .insightface/ /app/.insightface/
COPY artifacts_full/ /app/artifacts_full/
COPY deepfake_model.pt /app/deepfake_model.pt

# Copy backend application source code
COPY src/ /app/src/

# Copy compiled frontend from Stage 1 (FastAPI automatically serves SPA from /app/frontend/dist)
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Ensure persistent data directory exists for SQLite database & uploaded media
RUN mkdir -p /app/data

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8000}/api/health || exit 1

# Start FastAPI application
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

