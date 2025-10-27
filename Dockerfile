# DeepSeek-OCR RunPod Dockerfile
# Optimized for CUDA 11.8+ GPUs

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention (optional but recommended for speed)
RUN pip install flash-attn==2.7.3 --no-build-isolation || echo "Flash attention install failed, continuing without it"

# Copy handler script
COPY deepseek_handler.py .

# Pre-download the DeepSeek-OCR model during build
RUN python -c "\
import torch; \
from transformers import AutoModel, AutoTokenizer; \
print('Pre-downloading DeepSeek-OCR model...'); \
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True); \
model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True, use_safetensors=True); \
print('Model pre-downloaded successfully!')"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# RunPod will automatically call the handler
CMD ["python", "-u", "deepseek_handler.py"]

