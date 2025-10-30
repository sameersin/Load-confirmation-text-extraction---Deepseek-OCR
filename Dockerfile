# DeepSeek-OCR RunPod Dockerfile
# Optimized for CUDA 11.8+ GPUs

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies including poppler for PDF rendering
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention (optional but recommended for speed)
RUN pip install flash-attn==2.7.3 --no-build-isolation || echo "Flash attention install failed, continuing without it"

# Copy handler script
COPY deepseek_handler.py .

# Pre-download the DeepSeek-OCR model during build (matches handler loading)
RUN python -c "\
import torch; \
from transformers import AutoModel, AutoTokenizer; \
print('Pre-downloading DeepSeek-OCR model...'); \
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True); \
model = AutoModel.from_pretrained(\
    'deepseek-ai/DeepSeek-OCR', \
    trust_remote_code=True, \
    torch_dtype=torch.bfloat16, \
    device_map='auto' \
); \
print('Model pre-downloaded successfully!')"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# RunPod will automatically call the handler
CMD ["python", "-u", "deepseek_handler.py"]