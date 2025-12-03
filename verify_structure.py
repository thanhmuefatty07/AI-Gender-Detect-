# ============================================
# verify_structure.py
# Verify project structure is complete
# ============================================

import os
from pathlib import Path

def verify_structure():
    """Verify project structure is complete"""

    required_dirs = [
        "config", "data_collection", "datasets", "models",
        "preprocessing", "training", "evaluation", "inference",
        "app", "monitoring", "deployment", "scripts",
        "tests", "notebooks", "utils", "docs", "logs"
    ]

    print("🔍 Verifying project structure...")
    print("="*70)

    missing = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing.append(dir_name)
            print(f"❌ Missing: {dir_name}")
        else:
            # Count subdirectories
            subdirs = len([d for d in Path(dir_name).rglob("*") if d.is_dir()])
            print(f"✅ Found: {dir_name} ({subdirs} subdirs)")

    print("="*70)

    if missing:
        print(f"\n⚠️  {len(missing)} directories missing!")
        print("Run: bash setup_structure.sh")
    else:
        total_dirs = sum(len([d for d in Path(dir_name).rglob("*") if d.is_dir()])
                         for dir_name in required_dirs)
        print(f"\n✅ All {len(required_dirs)} main directories present!")
        print(f"📊 Total directories: {total_dirs}")
        print("\n🎉 Project structure complete!")

if __name__ == "__main__":
    verify_structure()

