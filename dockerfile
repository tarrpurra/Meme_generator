# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libcairo2 \
    libpango1.0-0 \
    libgobject-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production image
FROM python:3.11-slim

# Install only runtime dependencies for Pillow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libcairo2 \
    libpango1.0-0 \
    libgobject-2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/generated_images && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Make sure scripts are executable
RUN chmod +x /app/*.py

# Expose port
EXPOSE 8000

# Add local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Health check (using curl instead of requests for reliability)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Add environment variables for Railway
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Start server with Railway-optimized settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log", "--log-level", "warning", "--no-server-header"]
