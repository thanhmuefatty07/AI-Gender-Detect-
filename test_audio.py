#!/usr/bin/env python3
"""
Test Audio Model Inference

Test the trained audio model with audio files.
Supports gender and age prediction from voice.

Usage:
    python test_audio.py <audio_path>
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# ============================================
# Model Definition (MUST MATCH TRAINING)
# ============================================

class AudioModel(nn.Module):
    """Audio model for gender and age prediction from voice"""

    def __init__(self, config=None):
        super().__init__()

        # Default config if not provided
        if config is None:
            config = {
                'input_size': 40,  # MFCC features
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3
            }

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )

        # Gender classification head (binary)
        self.gender_head = nn.Sequential(
            nn.Linear(config['hidden_size'], 64),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Age regression head
        self.age_head = nn.Sequential(
            nn.Linear(config['hidden_size'], 64),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Forward pass"""
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]  # Shape: (batch, hidden_size)

        # Gender prediction
        gender = self.gender_head(last_hidden).squeeze(-1)

        # Age prediction
        age = self.age_head(last_hidden).squeeze(-1)

        return {
            'gender': gender,
            'age': age
        }

# ============================================
# Audio Preprocessing
# ============================================

def extract_mfcc_features(audio_path, n_mfcc=40, duration=3.0, sr=16000):
    """
    Extract MFCC features from audio file

    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        duration: Duration to extract (seconds)
        sr: Sample rate

    Returns:
        numpy array of shape (time_steps, n_mfcc)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)

        if len(y) == 0:
            print(f"❌ Audio file is empty: {audio_path}")
            return None

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            hop_length=512,
            n_fft=2048
        )

        # Transpose to (time_steps, features)
        mfcc = mfcc.T

        # Normalize
        mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)

        return mfcc

    except Exception as e:
        print(f"❌ Error extracting MFCC: {e}")
        return None

# ============================================
# Model Loading
# ============================================

def load_model(model_path):
    """Load trained audio model from checkpoint"""
    print("🎵 Loading Audio Model...")
    print("="*70)

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Make sure to run audio training first or download the model file.")
        return None

    # Initialize model
    model = AudioModel()

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
                print(f"   🏗️  Architecture: LSTM-{config.get('num_layers', 2)} layers")

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

def predict_single_audio(model, audio_path):
    """Predict gender and age from single audio file"""

    # Extract MFCC features
    print(f"🎵 Processing audio: {audio_path}")
    mfcc_features = extract_mfcc_features(audio_path)

    if mfcc_features is None:
        return None

    print(f"   📊 MFCC shape: {mfcc_features.shape}")
    print(f"   ⏱️  Duration: {mfcc_features.shape[0] * 0.032:.1f} seconds (estimated)")

    # Convert to tensor
    mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        outputs = model(mfcc_tensor)

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
        'audio_path': audio_path,
        'mfcc_shape': mfcc_features.shape
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

    print(f"\n🎵 AUDIO INFO:")
    print(f"   MFCC shape: {result['mfcc_shape']}")

    print("\n" + "="*70)

# ============================================
# Main Function
# ============================================

def main():
    """Main inference function"""
    print("🚀 Gender-Age Audio Classifier - Inference Mode")
    print("="*70)

    # Check arguments
    if len(sys.argv) < 2:
        print("\n📖 USAGE:")
        print("   python test_audio.py <audio_path>")
        print("\n📝 EXAMPLES:")
        print("   python test_audio.py voice.wav")
        print("   python test_audio.py ./audio_samples/speech.mp3")
        print("\n📋 REQUIREMENTS:")
        print("   - PyTorch installed")
        print("   - librosa installed")
        print("   - numpy installed")
        print("   - best_audio_model.pth in current directory")
        print("\n🔧 INSTALL DEPENDENCIES:")
        print("   pip install torch librosa numpy")
        print("\n📝 SUPPORTED FORMATS:")
        print("   - WAV, MP3, M4A, FLAC, OGG")
        sys.exit(0)

    audio_path = sys.argv[1]

    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)

    # Setup
    model_path = "models/best_audio_model.pth"

    # Load model
    model = load_model(model_path)
    if model is None:
        sys.exit(1)

    # Make prediction
    print(f"\n🔍 Analyzing: {audio_path}")
    print("-" * 70)

    result = predict_single_audio(model, audio_path)

    if result:
        print_results(result)
        print("✅ Inference complete!")
    else:
        print("❌ Inference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

