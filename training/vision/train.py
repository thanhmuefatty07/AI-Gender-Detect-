#!/usr/bin/env python3
"""
Vision Model Training Script
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import wandb
from datetime import datetime

from models.vision.architectures.efficientnet_model import EfficientNetModel
from preprocessing.face_detection.face_detector import FaceDetector
from utils.logging.logger import setup_logger
from utils.metrics.classification_metrics import ClassificationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_experiment(config, experiment_name):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"vision_{timestamp}"

    exp_dir = Path(config['training']['log_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    return exp_dir, exp_name


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metrics = ClassificationMetrics()

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        gender_labels = batch['gender'].to(device)
        age_labels = batch['age'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                gender_loss = criterion['gender'](outputs['gender'], gender_labels)
                age_loss = criterion['age'](outputs['age'], age_labels)
                loss = gender_loss + age_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            gender_loss = criterion['gender'](outputs['gender'], gender_labels)
            age_loss = criterion['age'](outputs['age'], age_labels)
            loss = gender_loss + age_loss

            loss.backward()
            optimizer.step()

        # Update metrics
        metrics.update(outputs, {'gender': gender_labels, 'age': age_labels})
        total_loss += loss.item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(train_loader), metrics.compute()


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    metrics = ClassificationMetrics()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            gender_labels = batch['gender'].to(device)
            age_labels = batch['age'].to(device)

            outputs = model(images)
            gender_loss = criterion['gender'](outputs['gender'], gender_labels)
            age_loss = criterion['age'](outputs['age'], age_labels)
            loss = gender_loss + age_loss

            metrics.update(outputs, {'gender': gender_labels, 'age': age_labels})
            total_loss += loss.item()

    return total_loss / len(val_loader), metrics.compute()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup
    exp_dir, exp_name = setup_experiment(config, args.experiment_name)
    logger = setup_logger("training", exp_dir / "train.log")
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')

    logger.info(f"Starting training experiment: {exp_name}")
    logger.info(f"Using device: {device}")

    # Initialize W&B if enabled
    if config.get('logging', {}).get('wandb', {}).get('enabled', False):
        wandb.init(
            project=config['logging']['wandb']['project'],
            name=exp_name,
            config=config
        )

    # Model
    model = EfficientNetModel(config['model'])
    model = model.to(device)

    # Multi-GPU training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")

    # Loss functions
    criterion = {
        'gender': nn.BCEWithLogitsLoss(),
        'age': nn.MSELoss()
    }

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config['hardware'].get('mixed_precision', False) else None

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        logger.info(f"Resumed from epoch {start_epoch}")

    # Data loaders (placeholder - implement actual data loading)
    # train_loader = get_train_loader(config)
    # val_loader = get_val_loader(config)

    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, None, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_metrics = validate(model, None, criterion, device)

        # Scheduler step
        scheduler.step()

        # Logging
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'lr': optimizer.param_groups[0]['lr']
        }

        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train Gender Acc: {train_metrics.get('gender_acc', 0):.4f}")
        logger.info(f"Val Gender Acc: {val_metrics.get('gender_acc', 0):.4f}")

        if config.get('logging', {}).get('wandb', {}).get('enabled', False):
            wandb.log(log_data)

        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = exp_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

        # Regular checkpoint
        if (epoch + 1) % config['training'].get('checkpointing', {}).get('save_frequency', 5) == 0:
            checkpoint_path = exp_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'config': config
            }, checkpoint_path)

    logger.info("Training completed!")
    if config.get('logging', {}).get('wandb', {}).get('enabled', False):
        wandb.finish()


if __name__ == "__main__":
    main()

