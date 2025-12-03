#!/usr/bin/env python3
"""
Model Converter Utility

Convert PyTorch models to ONNX format for production deployment.

Usage:
    python scripts/utils/model_converter.py input_model.pth output_model.onnx

Features:
- PyTorch to ONNX conversion
- Dynamic batch size support
- Model verification
- Metadata preservation
"""

import sys
import os
from pathlib import Path
import torch
import torch.onnx
import onnx
import numpy as np
from typing import Optional, Tuple
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelConverter:
    """Convert PyTorch models to ONNX format"""

    def __init__(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        """
        Initialize converter

        Args:
            input_shape: Model input shape (batch_size, channels, height, width)
        """
        self.input_shape = input_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_pytorch_model(self, model_path: str) -> Tuple[torch.nn.Module, dict]:
        """
        Load PyTorch model from checkpoint

        Args:
            model_path: Path to .pth checkpoint file

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        logger.info(f"Loading PyTorch model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Import your model architecture here
        # This is a placeholder - replace with your actual model
        try:
            from models.vision.architectures.efficientnet_model import EfficientNetModel

            # Get model config from checkpoint or use default
            model_config = checkpoint.get('config', {
                'architecture': 'efficientnet_b0',
                'pretrained': True,
                'dropout': 0.3
            })

            model = EfficientNetModel(model_config)

        except ImportError:
            logger.warning("Could not import model architecture. Using placeholder.")
            # Placeholder model for testing
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 2)  # gender + age
            )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        logger.info("✅ PyTorch model loaded successfully")
        return model, checkpoint

    def convert_to_onnx(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_sample: Optional[torch.Tensor] = None,
        opset_version: int = 14,
        verbose: bool = False
    ) -> None:
        """
        Convert PyTorch model to ONNX format

        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            input_sample: Sample input tensor
            opset_version: ONNX opset version
            verbose: Enable verbose logging
        """
        logger.info(f"Converting to ONNX: {output_path}")

        # Create dummy input if not provided
        if input_sample is None:
            input_sample = torch.randn(self.input_shape, device=self.device)

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        torch.onnx.export(
            model,
            input_sample,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['gender', 'age'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'gender': {0: 'batch_size'},
                'age': {0: 'batch_size'}
            },
            verbose=verbose
        )

        logger.info("✅ ONNX conversion completed")

    def verify_onnx_model(self, onnx_path: str) -> bool:
        """
        Verify ONNX model integrity

        Args:
            onnx_path: Path to ONNX model

        Returns:
            True if model is valid
        """
        logger.info(f"Verifying ONNX model: {onnx_path}")

        try:
            # Load model
            onnx_model = onnx.load(onnx_path)

            # Check model
            onnx.checker.check_model(onnx_model)

            # Print model info
            logger.info(f"Model inputs: {[inp.name for inp in onnx_model.graph.input]}")
            logger.info(f"Model outputs: {[out.name for out in onnx_model.graph.output]}")

            logger.info("✅ ONNX model verification passed")
            return True

        except Exception as e:
            logger.error(f"❌ ONNX model verification failed: {e}")
            return False

    def get_model_info(self, onnx_path: str) -> dict:
        """
        Get ONNX model information

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Dictionary with model information
        """
        onnx_model = onnx.load(onnx_path)

        info = {
            'opset_version': onnx_model.opset_import[0].version,
            'inputs': [
                {
                    'name': inp.name,
                    'shape': [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in inp.type.tensor_type.shape.dim]
                }
                for inp in onnx_model.graph.input
            ],
            'outputs': [
                {
                    'name': out.name,
                    'shape': [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in out.type.tensor_type.shape.dim]
                }
                for out in onnx_model.graph.output
            ],
            'num_nodes': len(onnx_model.graph.node),
            'num_params': sum(1 for _ in onnx_model.graph.initializer)
        }

        return info

    def convert(
        self,
        input_model: str,
        output_model: str,
        input_shape: Optional[Tuple[int, ...]] = None
    ) -> bool:
        """
        Complete conversion pipeline

        Args:
            input_model: Input PyTorch model path
            output_model: Output ONNX model path
            input_shape: Override input shape

        Returns:
            True if conversion successful
        """
        try:
            # Update input shape if provided
            if input_shape:
                self.input_shape = input_shape

            # Load PyTorch model
            model, checkpoint = self.load_pytorch_model(input_model)

            # Convert to ONNX
            self.convert_to_onnx(model, output_model)

            # Verify ONNX model
            if not self.verify_onnx_model(output_model):
                return False

            # Print model info
            info = self.get_model_info(output_model)
            logger.info("📊 Model Information:")
            for key, value in info.items():
                logger.info(f"   {key}: {value}")

            # Save metadata
            metadata = {
                'original_checkpoint': input_model,
                'onnx_path': output_model,
                'input_shape': list(self.input_shape),
                'converted_at': str(Path(output_model).stat().st_mtime),
                'model_info': info,
                'checkpoint_info': {
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'best_loss': checkpoint.get('best_loss', 'unknown'),
                    'config': checkpoint.get('config', {})
                }
            }

            metadata_path = output_model.replace('.onnx', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"💾 Metadata saved: {metadata_path}")
            logger.info("🎉 Conversion completed successfully!")

            return True

        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            return False


def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("input_model", help="Input PyTorch model (.pth)")
    parser.add_argument("output_model", help="Output ONNX model (.onnx)")
    parser.add_argument("--input-shape", nargs='+', type=int,
                       help="Input shape (e.g., 1 3 224 224)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(args.input_shape) if args.input_shape else None

    # Create converter
    converter = ModelConverter()

    # Convert model
    success = converter.convert(
        args.input_model,
        args.output_model,
        input_shape
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

