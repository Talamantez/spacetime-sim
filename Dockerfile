FROM python:3.11-slim

# Install system dependencies
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
RUN mkdir -p static/outputs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.main
ENV FLASK_ENV=production

# Print environment for debugging
CMD echo "Port is: $PORT" && \
    gunicorn \
    --bind "0.0.0.0:$PORT" \
    --timeout 300 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    app:app