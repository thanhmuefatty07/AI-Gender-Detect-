#!/usr/bin/env python3
"""
Simple Model File Checker

Check basic information about PyTorch model files without importing torch
"""

import os
from pathlib import Path
import time

def get_file_info(file_path):
    """Get basic file information"""
    stat = os.stat(file_path)

    return {
        'name': Path(file_path).name,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': time.ctime(stat.st_ctime),
        'modified': time.ctime(stat.st_mtime),
        'exists': True
    }

def check_file_header(file_path, num_bytes=100):
    """Check file header to identify file type"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(num_bytes)

        # Check for PyTorch magic number (PK\x03\x04 for ZIP, which PyTorch uses)
        if header.startswith(b'PK\x03\x04'):
            return "PyTorch ZIP archive (likely checkpoint)"

        # Check for other patterns
        header_hex = header.hex()
        if '504b' in header_hex:  # PK in hex
            return "ZIP archive (PyTorch format)"

        return f"Unknown format - starts with: {header[:20].hex()}"

    except Exception as e:
        return f"Error reading header: {e}"

def main():
    """Check all model files"""
    models_dir = Path("models")

    if not models_dir.exists():
        print("❌ Models directory not found!")
        return

    pth_files = list(models_dir.glob("*.pth"))

    if not pth_files:
        print("❌ No .pth files found!")
        return

    print("🔍 MODEL FILES ANALYSIS")
    print("=" * 60)

    for pth_file in sorted(pth_files):
        print(f"\n📁 File: {pth_file.name}")
        print("-" * 40)

        # Get file info
        info = get_file_info(pth_file)
        print(f"📊 Size: {info['size_mb']:.2f} MB ({info['size_bytes']:,} bytes)")
        print(f"📅 Created: {info['created']}")
        print(f"📅 Modified: {info['modified']}")

        # Check file header
        header_info = check_file_header(pth_file)
        print(f"🔍 Format: {header_info}")

        # Basic validation
        if info['size_bytes'] < 1000:
            print("⚠️  WARNING: File is very small (< 1KB)")
        elif info['size_bytes'] > 500 * 1024 * 1024:  # 500MB
            print("⚠️  WARNING: File is very large (> 500MB)")
        else:
            print("✅ Size looks reasonable")

    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"📁 Total model files: {len(pth_files)}")

    total_size = sum(get_file_info(f)['size_bytes'] for f in pth_files)
    print(f"💾 Total size: {total_size / (1024*1024):.2f} MB")
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Run full verification when PyTorch environment is available")
    print("2. Convert to ONNX for production deployment")
    print("3. Test inference with sample data")
    print("4. Backup models to cloud storage")

if __name__ == "__main__":
    main()
