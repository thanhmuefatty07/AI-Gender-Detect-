#!/usr/bin/env python3
"""
Model Verification Script

Verify PyTorch model files downloaded from Colab training.
Check file integrity, model structure, and compatibility.
"""

import torch
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def verify_pytorch_model(model_path):
    """
    Verify PyTorch model file

    Args:
        model_path: Path to .pth file

    Returns:
        dict: Verification results
    """
    results = {
        'file_exists': False,
        'file_size': 0,
        'is_valid_pytorch': False,
        'checkpoint_keys': [],
        'model_info': {},
        'errors': []
    }

    try:
        # Check if file exists
        if not os.path.exists(model_path):
            results['errors'].append(f"File not found: {model_path}")
            return results

        results['file_exists'] = True

        # Get file size
        file_size = os.path.getsize(model_path)
        results['file_size'] = file_size
        print(f"📁 File size: {file_size / (1024*1024):.2f} MB")

        # Try to load as PyTorch checkpoint
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            results['is_valid_pytorch'] = True

            # Analyze checkpoint structure
            if isinstance(checkpoint, dict):
                results['checkpoint_keys'] = list(checkpoint.keys())

                # Check for common PyTorch checkpoint keys
                if 'model_state_dict' in checkpoint:
                    print("✅ Standard PyTorch checkpoint format")
                    results['checkpoint_type'] = 'standard'

                    # Get model state dict info
                    state_dict = checkpoint['model_state_dict']
                    results['num_parameters'] = sum(p.numel() for p in state_dict.values())
                    results['model_layers'] = len(state_dict)

                    print(f"📊 Model parameters: {results['num_parameters']:,}")
                    print(f"🏗️  Model layers: {results['model_layers']}")

                elif 'state_dict' in checkpoint:
                    print("✅ Alternative checkpoint format")
                    results['checkpoint_type'] = 'alternative'
                else:
                    print("⚠️  Non-standard checkpoint format")
                    results['checkpoint_type'] = 'unknown'

                # Check for training metadata
                if 'epoch' in checkpoint:
                    results['epoch'] = checkpoint['epoch']
                    print(f"🎯 Epoch: {checkpoint['epoch']}")

                if 'best_loss' in checkpoint:
                    results['best_loss'] = checkpoint['best_loss']
                    print(f"📉 Best loss: {checkpoint['best_loss']:.4f}")

                if 'config' in checkpoint:
                    results['has_config'] = True
                    print("⚙️  Training config included")
                else:
                    results['has_config'] = False

            else:
                print("❌ Checkpoint is not a dictionary")
                results['errors'].append("Invalid checkpoint structure")

        except Exception as e:
            results['errors'].append(f"Failed to load PyTorch model: {str(e)}")
            print(f"❌ Error loading model: {e}")

    except Exception as e:
        results['errors'].append(f"Unexpected error: {str(e)}")
        print(f"💥 Unexpected error: {e}")

    return results

def analyze_model_architecture(model_path, results):
    """
    Analyze model architecture if possible

    Args:
        model_path: Path to model file
        results: Verification results dict
    """
    if not results['is_valid_pytorch']:
        return

    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

            # Analyze layer names
            layer_names = list(state_dict.keys())
            results['layer_names'] = layer_names[:10]  # First 10 layers

            # Check for common architectures
            if any('efficientnet' in name.lower() for name in layer_names):
                results['architecture'] = 'EfficientNet'
                print("🏗️  Architecture: EfficientNet")
            elif any('resnet' in name.lower() for name in layer_names):
                results['architecture'] = 'ResNet'
                print("🏗️  Architecture: ResNet")
            elif any('vgg' in name.lower() for name in layer_names):
                results['architecture'] = 'VGG'
                print("🏗️  Architecture: VGG")
            elif any('transformer' in name.lower() for name in layer_names):
                results['architecture'] = 'Transformer'
                print("🏗️  Architecture: Transformer")
            else:
                results['architecture'] = 'Unknown'
                print("🏗️  Architecture: Unknown/Custom")

            # Check for classification heads
            gender_layers = [name for name in layer_names if 'gender' in name.lower()]
            age_layers = [name for name in layer_names if 'age' in name.lower()]

            results['gender_layers'] = len(gender_layers)
            results['age_layers'] = len(age_layers)

            print(f"👤 Gender classification layers: {len(gender_layers)}")
            print(f"🎂 Age regression layers: {len(age_layers)}")

    except Exception as e:
        print(f"⚠️  Could not analyze architecture: {e}")

def main():
    """Main verification function"""
    print("🔍 MODEL VERIFICATION REPORT")
    print("=" * 50)

    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return

    # Find all .pth files
    pth_files = list(models_dir.glob("*.pth"))
    if not pth_files:
        print("❌ No .pth files found in models directory!")
        return

    print(f"📁 Found {len(pth_files)} model file(s):")
    for pth_file in pth_files:
        print(f"   • {pth_file.name}")
    print()

    # Verify each model
    all_results = {}
    for pth_file in pth_files:
        print(f"🔍 Verifying: {pth_file.name}")
        print("-" * 30)

        results = verify_pytorch_model(pth_file)
        analyze_model_architecture(pth_file, results)
        all_results[pth_file.name] = results

        # Summary for this model
        status = "✅ VALID" if results['is_valid_pytorch'] and not results['errors'] else "❌ INVALID"
        print(f"📋 Status: {status}")
        print()

    # Overall summary
    print("=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)

    valid_models = 0
    total_size = 0

    for model_name, results in all_results.items():
        status = "✅" if results['is_valid_pytorch'] and not results['errors'] else "❌"
        size_mb = results['file_size'] / (1024 * 1024)
        total_size += results['file_size']

        print(f"{status} {model_name}")
        print(f"   Size: {size_mb:.2f} MB")
        if results['is_valid_pytorch']:
            if 'num_parameters' in results:
                print(f"   Parameters: {results['num_parameters']:,}")
            if 'architecture' in results:
                print(f"   Architecture: {results['architecture']}")
            if 'epoch' in results:
                print(f"   Epoch: {results['epoch']}")
            valid_models += 1

        if results['errors']:
            print(f"   Errors: {len(results['errors'])}")
            for error in results['errors'][:2]:  # Show first 2 errors
                print(f"     - {error}")

        print()

    # Final assessment
    print("=" * 50)
    print("🎯 FINAL ASSESSMENT")
    print("=" * 50)

    print(f"📁 Total models: {len(all_results)}")
    print(f"✅ Valid models: {valid_models}")
    print(f"💾 Total size: {total_size / (1024*1024):.2f} MB")
    if valid_models == len(all_results):
        print("🎉 ALL MODELS ARE VALID AND READY FOR DEPLOYMENT!")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Convert to ONNX: python scripts/utils/model_converter.py best_vision_model.pth models/vision/exports/model.onnx")
        print("2. Test inference: python scripts/utils/quick_test.py models/vision/exports/model.onnx test_image.jpg")
        print("3. Deploy API: python inference/api/main.py")
    else:
        print("⚠️  SOME MODELS HAVE ISSUES - CHECK ERRORS ABOVE")

if __name__ == "__main__":
    main()
