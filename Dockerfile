FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    pillow \
    faiss-cpu \
    fastapi \
    uvicorn \
    python-multipart \
    scikit-learn \
    tqdm \
    sentencepiece

# Copy source code
COPY src /app/src

# Create directories for volumes
RUN mkdir -p /app/samples /app/index

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
