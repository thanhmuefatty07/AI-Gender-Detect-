#!/usr/bin/env python3
"""
System Test Script

Validates that all components are working correctly:
- Configuration loading
- Collector initialization
- Video processing pipeline
- Quality assessment
- Data validation
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test all imports work"""
    print("🔍 Testing imports...")

    try:
        # Base components
        from data_collection.base_collector import BaseCollector, CollectionMetadata
        print("✅ Base collector imports")

        # Collectors
        from data_collection.youtube_collector import YouTubeCollector
        print("✅ YouTube collector imports")

        from data_collection.tiktok_collector import TikTokCollector
        print("✅ TikTok collector imports")

        from data_collection.instagram_collector import InstagramCollector
        print("✅ Instagram collector imports")

        # Processing
        from data_collection.video_processor import VideoProcessor, FaceExtractor, QualityAssessor
        print("✅ Video processor imports")

        # Scripts
        from scripts.academic_datasets_merger import AcademicDatasetsMerger
        print("✅ Academic datasets merger imports")

        # Dashboard
        import app.monitoring_dashboard
        print("✅ Monitoring dashboard imports")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("🔧 Testing configuration...")

    try:
        import yaml
        config_path = project_root / "config" / "collector_config.yaml"

        if not config_path.exists():
            print("❌ Config file not found")
            return False

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ['sources', 'processing', 'output']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False

        print("✅ Configuration loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_base_collector():
    """Test base collector functionality"""
    print("🏗️ Testing base collector...")

    try:
        from data_collection.base_collector import BaseCollector, CollectionMetadata

        # Test metadata creation
        metadata = CollectionMetadata(
            source="test",
            item_id="test_123",
            url="https://example.com",
            title="Test Video",
            inferred_gender="male",
            quality_score=0.85
        )

        if metadata.to_dict()['quality_score'] != 0.85:
            print("❌ Metadata creation failed")
            return False

        print("✅ Base collector test passed")
        return True

    except Exception as e:
        print(f"❌ Base collector test failed: {e}")
        return False

def test_face_processing():
    """Test face detection and quality assessment"""
    print("🖼️ Testing face processing...")

    try:
        from data_collection.video_processor import FaceExtractor, QualityAssessor
        import numpy as np

        # Create mock config
        config = {
            'processing': {
                'face_detection': {
                    'method': 'mediapipe',
                    'min_detection_confidence': 0.5
                },
                'quality_filters': {
                    'blur_threshold': 100,
                    'brightness_range': [40, 220],
                    'min_face_size': 80
                }
            }
        }

        # Test quality assessor with mock image
        assessor = QualityAssessor(config)
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        quality = assessor.assess_quality(mock_image)

        if 'overall_score' not in quality:
            print("❌ Quality assessment failed")
            return False

        print("✅ Face processing test passed")
        return True

    except Exception as e:
        print(f"❌ Face processing test failed: {e}")
        return False

def test_academic_merger():
    """Test academic datasets merger"""
    print("📚 Testing academic datasets merger...")

    try:
        from scripts.academic_datasets_merger import AcademicDatasetsMerger

        merger = AcademicDatasetsMerger()

        # Test configuration loading
        if not hasattr(merger, 'datasets_config'):
            print("❌ Merger configuration failed")
            return False

        print("✅ Academic datasets merger test passed")
        return True

    except Exception as e:
        print(f"❌ Academic merger test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required directories exist"""
    print("📁 Testing directory structure...")

    required_dirs = [
        "config",
        "data_collection",
        "app",
        "scripts",
        "datasets/collected",
        "datasets/academic",
        "logs"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False

    print("✅ Directory structure test passed")
    return True

def generate_test_report(results):
    """Generate test report"""
    print("\n" + "="*50)
    print("📊 SYSTEM TEST REPORT")
    print("="*50)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print("-" * 50)
    print(f"OVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All systems operational!")
    else:
        print("⚠️ Some tests failed. Check logs for details.")

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'passed': passed,
        'total': total,
        'success_rate': passed / total
    }

    report_path = project_root / "logs" / f"system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Detailed report saved: {report_path}")

def main():
    """Run all system tests"""
    print("🚀 Starting Gender Classification System Tests")
    print("="*60)

    tests = {
        'imports': test_imports,
        'config': test_config,
        'base_collector': test_base_collector,
        'face_processing': test_face_processing,
        'academic_merger': test_academic_merger,
        'directory_structure': test_directory_structure
    }

    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    generate_test_report(results)

    # Exit with appropriate code
    success = all(results.values())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
