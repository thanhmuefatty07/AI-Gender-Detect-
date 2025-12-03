#!/usr/bin/env python3
"""
FastAPI Inference Server for Gender-Age Classification
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import time
import asyncio
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import yaml

from models.vision.architectures.efficientnet_model import EfficientNetModel
from preprocessing.face_detection.face_detector import FaceDetector
from utils.logging.logger import setup_logger
from utils.metrics.api_metrics import APIMetrics


# Global variables
model = None
face_detector = None
device = None
config = None
logger = None
metrics = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model, face_detector, device, config, logger, metrics

    # Startup
    logger.info("🚀 Starting Gender-Age Classification API")

    # Load config
    config_path = os.getenv("CONFIG_PATH", "config/deployment_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model_config = config.get('serving', {})
    model_path = model_config.get('model_path', 'models/vision/exports/model.onnx')

    try:
        # Load ONNX model for inference
        import onnxruntime as ort
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        model = ort.InferenceSession(model_path, providers=providers)
        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    # Initialize face detector
    face_detector = FaceDetector()

    # Initialize metrics
    metrics = APIMetrics()

    logger.info("🎯 API ready for inference")

    yield

    # Shutdown
    logger.info("🛑 Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Gender-Age Classification API",
    description="Multi-modal gender and age prediction from images",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api', {}).get('cors', {}).get('origins', ["*"]) if config else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logger
logger = setup_logger("api", Path("logs/api") / f"api_{time.strftime('%Y%m%d')}.log")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics"""
    return metrics.get_metrics()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Predict gender and age from image"""
    start_time = time.time()

    try:
        # Validate file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(400, "Only PNG and JPEG images are supported")

        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Detect faces
        faces = face_detector.detect_faces(np.array(image))

        if not faces:
            raise HTTPException(400, "No faces detected in the image")

        # Process first face (you can modify to handle multiple faces)
        face = faces[0]

        # Preprocess face
        face_tensor = preprocess_face(face, image)

        # Run inference
        result = run_inference(face_tensor)

        # Update metrics
        processing_time = time.time() - start_time
        metrics.update(processing_time, len(faces))

        # Log prediction
        logger.info(".2f")

        return {
            "predictions": result,
            "faces_detected": len(faces),
            "processing_time": processing_time,
            "model_version": config.get('serving', {}).get('model_version', '1.0.0')
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        processing_time = time.time() - start_time
        metrics.update_error(processing_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Batch prediction for multiple images"""
    start_time = time.time()
    results = []

    try:
        for file in files:
            # Similar processing as single prediction
            # (Implement batch processing logic here)
            pass

        processing_time = time.time() - start_time
        metrics.update_batch(len(files), processing_time)

        return {
            "results": results,
            "total_images": len(files),
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(500, f"Batch processing error: {str(e)}")


def preprocess_face(face_bbox, image):
    """Preprocess detected face for model input"""
    # Extract face region
    x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']

    # Add padding
    padding = 0.1
    x = max(0, int(x - w * padding))
    y = max(0, int(y - h * padding))
    w = min(image.width - x, int(w * (1 + 2 * padding)))
    h = min(image.height - y, int(h * (1 + 2 * padding)))

    # Crop face
    face_image = image.crop((x, y, x + w, y + h))

    # Resize to model input size
    target_size = (224, 224)  # Adjust based on your model
    face_image = face_image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to tensor
    face_array = np.array(face_image).astype(np.float32) / 255.0

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    face_array = (face_array - mean) / std

    # Transpose to CHW format
    face_tensor = torch.from_numpy(face_array.transpose(2, 0, 1)).unsqueeze(0)

    return face_tensor


def run_inference(face_tensor):
    """Run model inference"""
    try:
        # ONNX inference
        ort_inputs = {model.get_inputs()[0].name: face_tensor.numpy()}
        ort_outputs = model.run(None, ort_inputs)

        # Process outputs
        gender_logits = ort_outputs[0]
        age_pred = ort_outputs[1]

        # Convert to probabilities/confidence
        gender_prob = 1 / (1 + np.exp(-gender_logits[0][0]))  # Sigmoid
        gender = "female" if gender_prob > 0.5 else "male"
        age = float(age_pred[0][0])

        return {
            "gender": {
                "prediction": gender,
                "confidence": float(max(gender_prob, 1 - gender_prob))
            },
            "age": {
                "prediction": age,
                "confidence": 0.8  # Placeholder - implement proper confidence calculation
            }
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🎯 Gender-Age Classification API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("👋 Gender-Age Classification API stopped")


if __name__ == "__main__":
    # Get port from environment or config
    port = int(os.getenv("PORT", config.get('api', {}).get('port', 8000) if config else 8000))
    host = os.getenv("HOST", config.get('api', {}).get('host', "0.0.0.0") if config else "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        workers=config.get('api', {}).get('workers', 1) if config else 1,
        access_log=True
    )
