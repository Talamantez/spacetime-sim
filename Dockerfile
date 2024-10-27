# Use Python 3.11 instead of 3.9
FROM python:3.11-slim

# Install system dependencies including FFmpeg and OpenCV dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create the outputs directory
RUN mkdir -p static/outputs

# Make port 5000 available
EXPOSE 5000

# Define environment variable
ENV PORT=5000
ENV PYTHONPATH=/app

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]