FROM python:3.11-slim

WORKDIR /app

# Basic dependencies first
COPY requirements.txt .
RUN pip install flask gunicorn numpy matplotlib scipy

# Copy application files first
COPY app app/
COPY templates templates/
COPY static static/

# Create outputs directory
RUN mkdir -p static/outputs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Let Railway set the port
CMD gunicorn \
    --bind "0.0.0.0:${PORT:-8080}" \
    --timeout 300 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --log-level debug \
    app.main:app