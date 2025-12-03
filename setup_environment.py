#!/usr/bin/env python3
"""
Environment Setup Script

Setup development environment for Gender-Age Classifier.

Usage:
    python setup_environment.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse

def run_command(cmd, description="", check=True):
    """Run shell command with error handling"""
    print(f"🔧 {description}")
    print(f"   Command: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success"            return True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def setup_virtual_environment():
    """Setup Python virtual environment"""
    print("\n🐍 Setting up virtual environment...")

    # Check Python version
    python_version = sys.version_info
    print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 8):
        print("   ❌ Python 3.8+ required")
        return False

    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False

    # Activate virtual environment
    activate_cmd = "venv\\Scripts\\activate" if platform.system() == "Windows" else "source venv/bin/activate"
    print(f"   To activate: {activate_cmd}")

    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")

    # Activate venv and install
    pip_cmd = "venv\\Scripts\\pip.exe" if platform.system() == "Windows" else "venv/bin/pip"

    commands = [
        (f"{pip_cmd} install --upgrade pip", "Upgrading pip"),
        (f"{pip_cmd} install -r requirements.txt", "Installing main dependencies"),
        (f"{pip_cmd} install -e .[dev]", "Installing development dependencies")
    ]

    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False

    return success

def setup_git_hooks():
    """Setup Git hooks"""
    print("\n🔗 Setting up Git hooks...")

    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        # Copy pre-commit hook
        pre_commit_src = Path("scripts/hooks/pre-commit")
        pre_commit_dst = hooks_dir / "pre-commit"

        if pre_commit_src.exists():
            import shutil
            shutil.copy2(pre_commit_src, pre_commit_dst)
            pre_commit_dst.chmod(0o755)
            print("   ✅ Pre-commit hook installed")
        else:
            print("   ⚠️  Pre-commit hook template not found")

    return True

def verify_installation():
    """Verify installation"""
    print("\n🔍 Verifying installation...")

    # Activate venv and test imports
    python_cmd = "venv\\Scripts\\python.exe" if platform.system() == "Windows" else "venv/bin/python"

    test_commands = [
        (f"{python_cmd} -c \"import torch; print(f'PyTorch: {torch.__version__}')\"", "Testing PyTorch"),
        (f"{python_cmd} -c \"import cv2; print(f'OpenCV: {cv2.__version__}')\"", "Testing OpenCV"),
        (f"{python_cmd} -c \"import fastapi; print(f'FastAPI: {fastapi.__version__}')\"", "Testing FastAPI"),
        (f"{python_cmd} -c \"import streamlit; print(f'Streamlit: {streamlit.__version__}')\"", "Testing Streamlit"),
    ]

    success = True
    for cmd, desc in test_commands:
        if not run_command(cmd, desc, check=False):
            success = False

    return success

def create_env_file():
    """Create .env file from template"""
    print("\n📝 Setting up environment file...")

    template_path = Path("env.template")
    env_path = Path(".env")

    if template_path.exists() and not env_path.exists():
        import shutil
        shutil.copy2(template_path, env_path)
        print("   ✅ Created .env from template")
        print("   ⚠️  Please edit .env with your API keys")
        return True
    elif env_path.exists():
        print("   ✅ .env already exists")
        return True
    else:
        print("   ❌ env.template not found")
        return False

def setup_directories():
    """Setup project directories"""
    print("\n📁 Setting up project directories...")

    # Run setup script
    if Path("setup_structure.sh").exists():
        if run_command("bash setup_structure.sh", "Running setup_structure.sh"):
            return run_command("python verify_structure.py", "Verifying structure")
    else:
        print("   ⚠️  setup_structure.sh not found")
        return False

def print_next_steps():
    """Print next steps"""
    print("\n" + "="*60)
    print("🎉 ENVIRONMENT SETUP COMPLETE!")
    print("="*60)

    activate_cmd = "venv\\Scripts\\activate" if platform.system() == "Windows" else "source venv/bin/activate"

    print("\n🚀 Next Steps:")
    print(f"1. Activate environment: {activate_cmd}")
    print("2. Edit .env file with your API keys")
    print("3. Run: python scripts/run_monitoring.py")
    print("4. Start development!")

    print("\n📚 Useful Commands:")
    print("• Test collectors: python -m pytest tests/unit/test_collectors.py -v")
    print("• Start API: python inference/api/main.py")
    print("• View docs: http://localhost:8000/docs")
    print("• Monitor: http://localhost:8501")

    print("\n🔧 When training is done:")
    print("1. Download model from Colab")
    print("2. Convert: python scripts/utils/model_converter.py model.pth model.onnx")
    print("3. Test: python scripts/utils/quick_test.py model.onnx test.jpg")
    print("4. Deploy: python inference/api/main.py")

def main():
    """Main setup function"""
    print("🚀 Gender-Age Classifier Environment Setup")
    print("="*50)

    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the project root directory")
        print("   cd gender_age_classifier")
        sys.exit(1)

    steps = [
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Directories", setup_directories),
        ("Environment File", create_env_file),
        ("Git Hooks", setup_git_hooks),
        ("Verification", verify_installation),
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
        print("\n⚠️  Some steps failed. Please check the errors above.")
        print("You can run individual steps manually or fix issues and re-run.")

if __name__ == "__main__":
    main()

