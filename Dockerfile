FROM python:3.11-slim

# Install system dependencies including FFmpeg and OpenCV dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p static/outputs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/ || exit 1

# Use PORT environment variable with a default and more explicit binding
CMD gunicorn \
    --bind "0.0.0.0:${PORT:-5000}" \
    --timeout 300 \
    --workers 2 \
    --threads 4 \
    --worker-class gthread \
    --worker-tmp-dir /dev/shm \
    --log-level debug \
    --capture-output \
    --enable-stdio-inheritance \
    app:app