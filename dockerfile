# Base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install OS-level dependencies required by Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (main.py, generator scripts, etc.)
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start server with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
