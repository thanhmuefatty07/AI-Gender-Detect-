#!/usr/bin/env python3
"""
Test Vision Model Inference

Test the trained vision model with image files.
Supports gender and age prediction from face images.

Usage:
    python test_vision.py <image_path>
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# ============================================
# Model Definition (MUST MATCH TRAINING)
# ============================================

class VisionModel(nn.Module):
    """Vision model for gender and age prediction"""

    def __init__(self, config=None):
        super().__init__()

        # Default config if not provided
        if config is None:
            config = {
                'architecture': 'efficientnet_b0',
                'pretrained': False,
                'dropout': 0.3
            }

        # Backbone (CNN feature extractor)
        self.backbone = timm.create_model(
            config['architecture'],
            pretrained=config['pretrained'],
            num_classes=0  # Remove classification head
        )

        # Get backbone output features
        backbone_features = self.backbone.num_features

        # Gender classification head - EXACT MATCH WITH TRAINED MODEL
        # Manual layer creation to match trained model naming
        self.gender_head = nn.ModuleDict({
            '1': nn.Linear(backbone_features, 512),  # gender_head.1
            '2': nn.BatchNorm1d(512),                 # gender_head.2
            '5': nn.Linear(512, 1),                   # gender_head.5
            '6': nn.Sigmoid()                         # Not in trained model, for inference
        })

        # Age regression head - EXACT MATCH WITH TRAINED MODEL
        # Manual layer creation to match trained model naming
        self.age_head = nn.ModuleDict({
            '1': nn.Linear(backbone_features, 512),  # age_head.1
            '2': nn.BatchNorm1d(512),                 # age_head.2
            '5': nn.Linear(512, 1)                    # age_head.5
        })

    def forward(self, x):
        """Forward pass - matching trained model architecture"""
        # Extract features
        features = self.backbone(x)

        # Gender prediction - manual forward through layers
        gender = self.gender_head['1'](features)  # Linear
        gender = self.gender_head['2'](gender)    # BatchNorm
        gender = self.gender_head['5'](gender)    # Linear
        gender = self.gender_head['6'](gender)    # Sigmoid
        gender = gender.squeeze(-1)

        # Age prediction - manual forward through layers
        age = self.age_head['1'](features)  # Linear
        age = self.age_head['2'](age)       # BatchNorm
        age = self.age_head['5'](age)       # Linear
        age = age.squeeze(-1)

        return {
            'gender': gender,
            'age': age
        }

# ============================================
# Preprocessing
# ============================================

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================
# Model Loading
# ============================================

def load_model(model_path):
    """Load trained model from checkpoint"""
    print("🤖 Loading Vision Model...")
    print("="*70)

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Make sure to run training first or download the model file.")
        return None

    # Initialize model
    model = VisionModel()

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)

        model.eval()

        print("✅ Model loaded successfully!")

        # Print training info if available
        if 'epoch' in checkpoint:
            print(f"   🏃 Trained for: {checkpoint['epoch']} epochs")

        if 'best_loss' in checkpoint:
            print(f"   📉 Best validation loss: {checkpoint['best_loss']:.4f}")

        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
            if 'gender_acc' in metrics:
                print(f"   👤 Gender accuracy: {metrics['gender_acc']*100:.2f}%")
            if 'age_mae' in metrics:
                print(f"   🎂 Age MAE: {metrics['age_mae']:.2f} years")

        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'architecture' in config:
                print(f"   🏗️  Architecture: {config['architecture']}")

        print("="*70)
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Possible issues:")
        print("  - Model architecture mismatch")
        print("  - Corrupted checkpoint file")
        print("  - Different PyTorch version")
        return None

# ============================================
# Inference Functions
# ============================================

def predict_single_image(model, image_path, transform):
    """Predict gender and age from single image"""

    # Load and validate image
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None

        image = Image.open(image_path).convert('RGB')
        print(f"📸 Image loaded: {image.size[0]}x{image.size[1]} pixels")

    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

    # Preprocess
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)

        # Get predictions
        gender_prob = outputs['gender'].item()
        age_value = outputs['age'].item()

    # Interpret results
    gender = "Female" if gender_prob > 0.5 else "Male"
    confidence = max(gender_prob, 1 - gender_prob)  # Confidence in prediction
    age = max(0, min(120, int(round(age_value))))  # Clamp age to reasonable range

    return {
        'gender': gender,
        'gender_confidence': confidence,
        'age': age,
        'raw_gender_prob': gender_prob,
        'raw_age_value': age_value,
        'image_path': image_path
    }

def print_results(result):
    """Pretty print prediction results"""
    if result is None:
        return

    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   👤 Gender: {result['gender']}")
    print(f"   📊 Confidence: {result['gender_confidence']*100:.1f}%")
    print(f"   🎂 Age: {result['age']} years old")

    print(f"\n📈 RAW VALUES:")
    print(f"   Gender probability: {result['raw_gender_prob']:.4f} (0=Male, 1=Female)")
    print(f"   Age regression: {result['raw_age_value']:.2f}")

    print("\n" + "="*70)

# ============================================
# Main Function
# ============================================

def main():
    """Main inference function"""
    print("🚀 Gender-Age Vision Classifier - Inference Mode")
    print("="*70)

    # Check arguments
    if len(sys.argv) < 2:
        print("\n📖 USAGE:")
        print("   python test_vision.py <image_path>")
        print("\n📝 EXAMPLES:")
        print("   python test_vision.py photo.jpg")
        print("   python test_vision.py ./test_images/face.png")
        print("\n📋 REQUIREMENTS:")
        print("   - PyTorch installed")
        print("   - Pillow (PIL) installed")
        print("   - timm installed")
        print("   - best_vision_model.pth in current directory")
        print("\n🔧 INSTALL DEPENDENCIES:")
        print("   pip install torch torchvision timm pillow")
        sys.exit(0)

    image_path = sys.argv[1]

    # Setup
    transform = get_transforms()
    model_path = "models/best_vision_model.pth"

    # Load model
    model = load_model(model_path)
    if model is None:
        sys.exit(1)

    # Make prediction
    print(f"\n🔍 Analyzing: {image_path}")
    print("-" * 70)

    result = predict_single_image(model, image_path, transform)

    if result:
        print_results(result)
        print("✅ Inference complete!")
    else:
        print("❌ Inference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

