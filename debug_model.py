#!/usr/bin/env python3
"""
Debug Model Architecture Mapping

Create a test model and see how PyTorch names layers in Sequential
"""

import torch
import torch.nn as nn

class TestVisionModel(nn.Module):
    """Test model to understand layer naming"""

    def __init__(self):
        super().__init__()

        # Simple backbone (mock)
        self.backbone = nn.Linear(10, 1280)

        # Gender head - same as trained model
        self.gender_head = nn.Sequential(
            nn.Linear(1280, 512),  # This should be gender_head.1
            nn.BatchNorm1d(512),   # This should be gender_head.2
            nn.Linear(512, 1),     # This should be gender_head.5 ?????
            nn.Sigmoid()
        )

        # Age head - same as trained model
        self.age_head = nn.Sequential(
            nn.Linear(1280, 512),  # This should be age_head.1
            nn.BatchNorm1d(512),   # This should be age_head.2
            nn.Linear(512, 1)      # This should be age_head.5 ?????
        )

    def forward(self, x):
        features = self.backbone(x)
        gender = self.gender_head(features)
        age = self.age_head(features)
        return {'gender': gender, 'age': age}

def analyze_layer_names():
    """Analyze how PyTorch names layers in state_dict"""

    print("🔍 DEBUGGING MODEL ARCHITECTURE")
    print("="*70)

    # Create test model
    model = TestVisionModel()

    # Get state dict
    state_dict = model.state_dict()

    print("📋 State Dict Keys and Shapes:")
    print("-" * 50)

    for name, param in state_dict.items():
        print(f"   {name}: {param.shape}")

    print("\n🏗️  Gender Head Analysis:")
    print("-" * 30)

    gender_layers = {k: v for k, v in state_dict.items() if k.startswith('gender_head')}
    for name, param in gender_layers.items():
        print(f"   {name}: {param.shape}")

    print("\n🏗️  Age Head Analysis:")
    print("-" * 30)

    age_layers = {k: v for k, v in state_dict.items() if k.startswith('age_head')}
    for name, param in age_layers.items():
        print(f"   {name}: {param.shape}")

    print("\n🔍 Expected from Trained Model:")
    print("-" * 35)
    print("   gender_head.1.weight: torch.Size([512, 1280])")
    print("   gender_head.2.* : BatchNorm parameters")
    print("   gender_head.5.weight: torch.Size([1, 512])")
    print()
    print("   age_head.1.weight: torch.Size([512, 1280])")
    print("   age_head.2.* : BatchNorm parameters")
    print("   age_head.5.weight: torch.Size([1, 512])")

    print("\n💡 INSIGHT:")
    print("-" * 10)
    print("The trained model uses different layer indices!")
    print("Sequential layers are named .0, .1, .2, .3 but")
    print("trained model has .1, .2, .5 - maybe custom naming?")

    return state_dict

def compare_with_trained_model():
    """Compare with actual trained model state dict"""

    print("\n🔄 COMPARING WITH TRAINED MODEL")
    print("="*70)

    try:
        # Load trained model
        trained_checkpoint = torch.load('models/best_vision_model.pth', map_location='cpu')
        trained_state = trained_checkpoint['model_state_dict']

        print("📋 Trained Model Gender Head Layers:")
        print("-" * 40)

        gender_layers = {k: v.shape for k, v in trained_state.items() if k.startswith('gender_head')}
        for name, shape in gender_layers.items():
            print(f"   {name}: {shape}")

        print("\n📋 Trained Model Age Head Layers:")
        print("-" * 40)

        age_layers = {k: v.shape for k, v in trained_state.items() if k.startswith('age_head')}
        for name, shape in age_layers.items():
            print(f"   {name}: {shape}")

    except Exception as e:
        print(f"❌ Could not load trained model: {e}")

if __name__ == "__main__":
    # Analyze layer naming
    test_state = analyze_layer_names()

    # Compare with trained model
    compare_with_trained_model()

    print("\n🎯 CONCLUSION:")
    print("-" * 15)
    print("Need to match EXACT layer names from trained model")
    print("The trained model likely has different Sequential structure")
