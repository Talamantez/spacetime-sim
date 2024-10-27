FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install flask gunicorn
COPY . .

CMD gunicorn --bind "0.0.0.0:$PORT" test:app