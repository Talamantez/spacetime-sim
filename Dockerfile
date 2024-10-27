FROM python:3.11-slim

WORKDIR /app

# Basic dependencies first
COPY requirements.txt .
RUN pip install flask gunicorn numpy matplotlib scipy

COPY . .
RUN mkdir -p static/outputs

ENV PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD gunicorn --bind "0.0.0.0:8080" \
    --timeout 300 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    app:app