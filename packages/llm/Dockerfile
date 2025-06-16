# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    PORT=8000 \
    WORKERS=1 \
    MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/

# Download the model during build (comment out for faster builds during development)
# RUN python -c "from sentence_transformers import SentenceTransformer; \
#     SentenceTransformer('$MODEL_NAME', cache_folder='$MODEL_DIR')"

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run the application
CMD exec uvicorn \
    --host 0.0.0.0 \
    --port $PORT \
    --workers $WORKERS \
    --log-level info \
    --no-access-log \
    src.main:app