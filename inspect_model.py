#!/usr/bin/env python3
"""
Inspect trained model architecture and structure
"""

import torch
import sys
import os

def inspect_checkpoint(checkpoint_path):
    """Inspect PyTorch checkpoint without loading model architecture"""

    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
    print("="*70)

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print("📋 Checkpoint Keys:")
        for key in checkpoint.keys():
            if key != 'model_state_dict':
                print(f"   • {key}: {type(checkpoint[key])}")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n🏗️  Model State Dict: {len(state_dict)} parameters")

            # Group layers by type
            layer_types = {}
            layer_shapes = {}

            for name, param in state_dict.items():
                # Get layer type from name
                layer_type = name.split('.')[0] if '.' in name else name

                if layer_type not in layer_types:
                    layer_types[layer_type] = []
                layer_types[layer_type].append(name)

                # Store shape info
                layer_shapes[name] = param.shape

            print("\n📊 Layer Types Found:")
            for layer_type, layers in layer_types.items():
                print(f"   • {layer_type}: {len(layers)} layers")

            print("\n🔍 Key Layer Shapes (first 10):")
            for i, (name, shape) in enumerate(layer_shapes.items()):
                if i >= 10:
                    break
                print(f"   • {name}: {shape}")

            # Check for BatchNorm
            has_batchnorm = any('running_mean' in name or 'running_var' in name for name in state_dict.keys())
            print(f"\n🔧 Has BatchNorm: {has_batchnorm}")

            # Check for Dropout (harder to detect)
            has_dropout = any('dropout' in name.lower() for name in state_dict.keys())
            print(f"🔧 Has Dropout: {has_dropout}")

        else:
            print("❌ No model_state_dict found in checkpoint")

        # Print training info if available
        if 'epoch' in checkpoint:
            print(f"\n🏃 Training Info:")
            print(f"   • Epoch: {checkpoint['epoch']}")

        if 'best_loss' in checkpoint:
            print(f"   • Best Loss: {checkpoint['best_loss']:.4f}")

        if 'val_metrics' in checkpoint:
            print(f"   • Validation Metrics: {checkpoint['val_metrics']}")

        if 'config' in checkpoint:
            print(f"   • Training Config: {checkpoint['config']}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")
        print("Possible issues:")
        print("  - Corrupted checkpoint file")
        print("  - Incompatible PyTorch version")
        return False

    return True

def create_compatible_model_class(state_dict):
    """Create a model class that matches the checkpoint architecture"""

    # Analyze the state_dict to understand architecture
    layer_info = {}

    for name, param in state_dict.items():
        parts = name.split('.')
        if len(parts) >= 2:
            layer_name = parts[0]
            if layer_name not in layer_info:
                layer_info[layer_name] = {'params': [], 'shapes': []}
            layer_info[layer_name]['params'].append(name)
            layer_info[layer_name]['shapes'].append(param.shape)

    print("🔧 Detected Layer Architecture:")
    for layer_name, info in layer_info.items():
        print(f"   • {layer_name}: {len(info['params'])} parameters")
        if info['shapes']:
            print(f"     Shapes: {info['shapes'][:3]}...")

    # Try to infer architecture
    if 'backbone' in layer_info:
        backbone_shapes = [shape for name, shape in zip(layer_info['backbone']['params'], layer_info['backbone']['shapes']) if 'weight' in name]
        if backbone_shapes:
            print(f"   📊 Backbone output features: {backbone_shapes[-1][0]}")

    # Gender head analysis
    if 'gender_head' in layer_info:
        gender_params = layer_info['gender_head']['params']
        gender_shapes = layer_info['gender_head']['shapes']

        print(f"   👤 Gender head: {len(gender_params)} layers")
        for i, (name, shape) in enumerate(zip(gender_params, gender_shapes)):
            print(f"     Layer {i}: {name} -> {shape}")

    # Age head analysis
    if 'age_head' in layer_info:
        age_params = layer_info['age_head']['params']
        age_shapes = layer_info['age_head']['shapes']

        print(f"   🎂 Age head: {len(age_params)} layers")
        for i, (name, shape) in enumerate(zip(age_params, age_shapes)):
            print(f"     Layer {i}: {name} -> {shape}")

def main():
    """Main inspection function"""
    print("🔬 MODEL ARCHITECTURE INSPECTOR")
    print("="*70)

    # Check all model files
    model_files = [
        'models/best_vision_model.pth',
        'models/best_audio_model.pth',
        'models/checkpoint_epoch_10.pth'
    ]

    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\n📁 Inspecting: {model_file}")
            print("-" * 50)

            if inspect_checkpoint(model_file):
                try:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        create_compatible_model_class(checkpoint['model_state_dict'])
                except Exception as e:
                    print(f"❌ Could not analyze architecture: {e}")
            else:
                print("❌ Inspection failed")
        else:
            print(f"❌ File not found: {model_file}")

    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("1. Update model architecture in test scripts to match")
    print("2. Use the layer information above to define correct model")
    print("3. Check training config for exact architecture used")

if __name__ == "__main__":
    main()
