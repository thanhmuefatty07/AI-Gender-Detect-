# ============================================
# Multi-stage Dockerfile for Gender-Age Classifier
# ============================================

# Base stage with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Development stage
# ============================================
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    black \
    flake8 \
    mypy \
    jupyter

# Copy source code
COPY . .

# Change ownership
RUN chown -R app:app /app
USER app

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ============================================
# Training stage
# ============================================
FROM base as training

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    wandb \
    tensorboard

# Copy source code
COPY . .

# Change ownership
RUN chown -R app:app /app
USER app

# Default command for training
CMD ["python", "training/vision/train.py"]

# ============================================
# Inference stage
# ============================================
FROM base as inference

# Install inference-specific dependencies
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    prometheus-client

# Copy source code
COPY . .

# Create model directory
RUN mkdir -p models/vision/exports

# Change ownership
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "inference/api/main.py"]

# ============================================
# Monitoring stage
# ============================================
FROM base as monitoring

# Install monitoring dependencies
RUN pip install --no-cache-dir \
    streamlit \
    plotly \
    prometheus-client

# Copy source code
COPY . .

# Change ownership
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app/monitoring_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

