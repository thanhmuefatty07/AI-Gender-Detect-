#!/usr/bin/env python3
"""
Setup Inference Environment

Install required dependencies for running inference with trained models.
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run shell command"""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version < (3, 8):
        print("   ❌ Python 3.8+ required")
        return False

    print("   ✅ Python version OK")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")

    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0"
    ]

    # Install packages
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False

    return True

def verify_installation():
    """Verify installations"""
    print("\n🔍 Verifying installations...")

    imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "PyTorch Image Models"),
        ("PIL", "Pillow"),
        ("librosa", "Librosa"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib")
    ]

    success = True
    for module, name in imports:
        try:
            if module == "PIL":
                import PIL
            else:
                __import__(module)
            print(f"   ✅ {name} imported successfully")
        except ImportError as e:
            print(f"   ❌ {name} import failed: {e}")
            success = False

    return success

def create_test_directories():
    """Create test directories"""
    print("\n📁 Creating test directories...")

    directories = [
        "test_images",
        "test_audio",
        "results"
    ]

    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ Created {dir_name}/")

def download_sample_data():
    """Download sample test data (optional)"""
    print("\n📥 Downloading sample data...")

    # This would download sample images/audio for testing
    # For now, just create placeholder files

    sample_image = "test_images/sample_face.jpg"
    sample_audio = "test_audio/sample_voice.wav"

    # Create placeholder files with instructions
    with open(sample_image, 'w') as f:
        f.write("# PLACE YOUR TEST IMAGE HERE\n")
        f.write("# Example: python test_vision.py test_images/sample_face.jpg\n")

    with open(sample_audio, 'w') as f:
        f.write("# PLACE YOUR TEST AUDIO HERE\n")
        f.write("# Example: python test_audio.py test_audio/sample_voice.wav\n")

    print("   ✅ Created sample data placeholders")
    print("   📝 Add your own test files to test_images/ and test_audio/")

def print_next_steps():
    """Print next steps"""
    print("\n" + "="*60)
    print("🎉 INFERENCE ENVIRONMENT SETUP COMPLETE!")
    print("="*60)

    print("\n🚀 NEXT STEPS:")
    print("1. Add test images to test_images/ folder")
    print("2. Add test audio to test_audio/ folder")
    print("3. Test vision model: python test_vision.py test_images/sample.jpg")
    print("4. Test audio model: python test_audio.py test_audio/sample.wav")

    print("\n📚 AVAILABLE SCRIPTS:")
    print("• test_vision.py - Test vision model")
    print("• test_audio.py - Test audio model")
    print("• setup_inference.py - Re-run this setup")

    print("\n🆘 TROUBLESHOOTING:")
    print("• If models fail to load, check file paths")
    print("• If imports fail, run: pip install -r requirements.txt")
    print("• For GPU support, install CUDA version of PyTorch")

def main():
    """Main setup function"""
    print("🚀 Gender-Age Classifier - Inference Setup")
    print("="*50)

    steps = [
        ("Check Python Version", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Verify Installation", verify_installation),
        ("Create Directories", create_test_directories),
        ("Setup Sample Data", download_sample_data),
    ]

    success_count = 0
    for step_name, step_func in steps:
        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name}: PASSED")
            else:
                print(f"❌ {step_name}: FAILED")
        except Exception as e:
            print(f"❌ {step_name}: ERROR - {e}")

    print(f"\n📊 Setup Results: {success_count}/{len(steps)} steps passed")

    if success_count == len(steps):
        print_next_steps()
    else:
        print("\n⚠️  Some steps failed. Check the errors above.")
        print("You can re-run this script or install dependencies manually.")

if __name__ == "__main__":
    main()

