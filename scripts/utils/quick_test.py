#!/usr/bin/env python3
"""
Quick Inference Test Utility

Test ONNX model inference with sample images.

Usage:
    python scripts/utils/quick_test.py model.onnx image.jpg

Features:
- ONNX model inference testing
- Image preprocessing
- Result visualization
- Performance benchmarking
- Batch testing support
"""

import sys
import os
from pathlib import Path
import time
import argparse
import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime not available. Install with: pip install onnxruntime-gpu")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InferenceTester:
    """Test ONNX model inference"""

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize tester

        Args:
            model_path: Path to ONNX model
            providers: ONNX execution providers
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not available")

        self.model_path = model_path

        # Setup providers
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']

        logger.info(f"Loading ONNX model: {model_path}")
        logger.info(f"Using providers: {providers}")

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"Input: {self.input_name}")
        logger.info(f"Outputs: {self.output_names}")

    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model input

        Args:
            image_path: Path to image file
            target_size: Target image size (height, width)

        Returns:
            Preprocessed image array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std

        # Transpose to CHW format
        image_array = np.transpose(image_array, (2, 0, 1))

        # Add batch dimension
        image_array = np.expand_dims(image_array, 0)

        return image_array

    def run_inference(self, image_array: np.ndarray) -> dict:
        """
        Run model inference

        Args:
            image_array: Preprocessed image array

        Returns:
            Dictionary with predictions
        """
        # Prepare inputs
        inputs = {self.input_name: image_array}

        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, inputs)
        inference_time = time.time() - start_time

        # Process outputs
        results = {}

        # Gender prediction (assuming sigmoid output)
        if len(outputs) >= 1:
            gender_logits = outputs[0]
            gender_prob = 1 / (1 + np.exp(-gender_logits[0][0]))  # Sigmoid
            gender = "Female" if gender_prob > 0.5 else "Male"
            gender_confidence = float(max(gender_prob, 1 - gender_prob))

            results['gender'] = {
                'prediction': gender,
                'confidence': gender_confidence,
                'probability': float(gender_prob)
            }

        # Age prediction (assuming regression output)
        if len(outputs) >= 2:
            age_pred = float(outputs[1][0][0])
            results['age'] = {
                'prediction': age_pred,
                'rounded': round(age_pred)
            }

        results['inference_time'] = inference_time

        return results

    def test_single_image(self, image_path: str, show_image: bool = True) -> dict:
        """
        Test inference on single image

        Args:
            image_path: Path to test image
            show_image: Whether to display image

        Returns:
            Test results dictionary
        """
        logger.info(f"Testing image: {image_path}")

        # Preprocess
        image_array = self.preprocess_image(image_path)

        # Run inference
        results = self.run_inference(image_array)

        # Display results
        self._display_results(results, image_path if show_image else None)

        return results

    def test_batch(self, image_paths: List[str], batch_size: int = 8) -> List[dict]:
        """
        Test inference on batch of images

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing

        Returns:
            List of test results
        """
        logger.info(f"Testing batch of {len(image_paths)} images")

        all_results = []
        inference_times = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_paths)} images")

            batch_images = []
            valid_paths = []

            # Preprocess batch
            for img_path in batch_paths:
                try:
                    img_array = self.preprocess_image(img_path)
                    batch_images.append(img_array)
                    valid_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Skipping {img_path}: {e}")

            if not batch_images:
                continue

            # Stack batch
            batch_array = np.concatenate(batch_images, axis=0)

            # Run inference
            inputs = {self.input_name: batch_array}
            start_time = time.time()
            outputs = self.session.run(self.output_names, inputs)
            batch_time = time.time() - start_time

            # Process results
            for j, img_path in enumerate(valid_paths):
                results = self._process_batch_output(outputs, j)
                results['image_path'] = img_path
                results['batch_inference_time'] = batch_time / len(valid_paths)
                inference_times.append(results['batch_inference_time'])

                all_results.append(results)

        # Summary statistics
        if inference_times:
            logger.info(f"📊 Batch Statistics:")
            logger.info(f"   Average inference time: {np.mean(inference_times)*1000:.2f}ms")
            logger.info(f"   Min time: {np.min(inference_times)*1000:.2f}ms")
            logger.info(f"   Max time: {np.max(inference_times)*1000:.2f}ms")

        return all_results

    def _process_batch_output(self, outputs: List[np.ndarray], index: int) -> dict:
        """Process batch output for single sample"""
        results = {}

        # Gender
        if len(outputs) >= 1:
            gender_logits = outputs[0][index]
            gender_prob = 1 / (1 + np.exp(-gender_logits[0]))  # Sigmoid
            gender = "Female" if gender_prob > 0.5 else "Male"
            gender_confidence = float(max(gender_prob, 1 - gender_prob))

            results['gender'] = {
                'prediction': gender,
                'confidence': gender_confidence,
                'probability': float(gender_prob)
            }

        # Age
        if len(outputs) >= 2:
            age_pred = float(outputs[1][index][0])
            results['age'] = {
                'prediction': age_pred,
                'rounded': round(age_pred)
            }

        return results

    def benchmark(self, image_path: str, num_runs: int = 100) -> dict:
        """
        Benchmark model performance

        Args:
            image_path: Path to test image
            num_runs: Number of inference runs

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking with {num_runs} runs...")

        # Preprocess once
        image_array = self.preprocess_image(image_path)

        # Warmup
        for _ in range(5):
            self.run_inference(image_array)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.run_inference(image_array)
            times.append(time.time() - start_time)

        times_ms = np.array(times) * 1000

        benchmark_results = {
            'num_runs': num_runs,
            'avg_time_ms': float(np.mean(times_ms)),
            'std_time_ms': float(np.std(times_ms)),
            'min_time_ms': float(np.min(times_ms)),
            'max_time_ms': float(np.max(times_ms)),
            'p50_time_ms': float(np.percentile(times_ms, 50)),
            'p95_time_ms': float(np.percentile(times_ms, 95)),
            'p99_time_ms': float(np.percentile(times_ms, 99)),
            'throughput': float(1000 / np.mean(times_ms))  # inferences per second
        }

        logger.info("📊 Benchmark Results:")
        logger.info(f"   Average: {benchmark_results['avg_time_ms']:.2f}ms")
        logger.info(f"   P95: {benchmark_results['p95_time_ms']:.2f}ms")
        logger.info(f"   Throughput: {benchmark_results['throughput']:.1f} inf/s")

        return benchmark_results

    def _display_results(self, results: dict, image_path: Optional[str] = None):
        """Display test results"""
        print("\n" + "="*50)
        print("🎯 INFERENCE RESULTS")
        print("="*50)

        if 'gender' in results:
            gender = results['gender']
            print(f"👤 Gender: {gender['prediction']} ({gender['confidence']*100:.1f}% confidence)")

        if 'age' in results:
            age = results['age']
            print(f"🎂 Age: {age['prediction']:.1f} years (rounded: {age['rounded']})")

        if 'inference_time' in results:
            print(f"⚡ Inference time: {results['inference_time']*1000:.2f}ms")

        print("="*50)


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument("model", help="ONNX model path")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("--batch", action="store_true", help="Test all images in directory")
    parser.add_argument("--benchmark", type=int, help="Run benchmark with N runs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for directory testing")
    parser.add_argument("--no-display", action="store_true", help="Don't display images")

    args = parser.parse_args()

    try:
        # Initialize tester
        tester = InferenceTester(args.model)

        if args.benchmark:
            # Benchmark mode
            if os.path.isfile(args.input):
                results = tester.benchmark(args.input, args.benchmark)

                # Save results
                import json
                benchmark_file = f"benchmark_results_{int(time.time())}.json"
                with open(benchmark_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Benchmark results saved: {benchmark_file}")

            else:
                logger.error("Benchmark mode requires a single image file")

        elif args.batch or os.path.isdir(args.input):
            # Batch mode
            if os.path.isdir(args.input):
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_paths = []

                for ext in image_extensions:
                    image_paths.extend(Path(args.input).glob(f"**/*{ext}"))

                image_paths = [str(p) for p in image_paths[:50]]  # Limit to 50 images

                if not image_paths:
                    logger.error(f"No images found in directory: {args.input}")
                    return

                logger.info(f"Found {len(image_paths)} images")

            else:
                logger.error("Batch mode requires a directory path")
                return

            # Run batch test
            results = tester.test_batch(image_paths, args.batch_size)

            # Summary
            if results:
                gender_counts = {}
                age_ranges = {'18-25': 0, '26-35': 0, '36-50': 0, '51+': 0}

                for result in results:
                    if 'gender' in result:
                        gender = result['gender']['prediction']
                        gender_counts[gender] = gender_counts.get(gender, 0) + 1

                    if 'age' in result:
                        age = result['age']['rounded']
                        if age <= 25:
                            age_ranges['18-25'] += 1
                        elif age <= 35:
                            age_ranges['26-35'] += 1
                        elif age <= 50:
                            age_ranges['36-50'] += 1
                        else:
                            age_ranges['51+'] += 1

                print("
📊 Batch Summary:"                print(f"   Gender distribution: {gender_counts}")
                print(f"   Age distribution: {age_ranges}")

        else:
            # Single image mode
            results = tester.test_single_image(args.input, not args.no_display)

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

