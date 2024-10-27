# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies including FFmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

# Run the application
CMD gunicorn app:app --bind 0.0.0.0:$PORT