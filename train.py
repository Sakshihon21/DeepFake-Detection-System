"""Main training script."""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils.config import load_config
from src.data.dataset import DeepFakeDataset, VideoDataset
from src.models import CNNModel, ViTDeepFakeModel, TemporalModel, EnsembleModel
from src.training import Trainer


def build_model(config):
    """Build model based on config."""
    model_config = config['model']
    model_name = model_config['name']
    
    if model_name == 'cnn':
        model = CNNModel(
            backbone=model_config['cnn_backbone'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            use_pretrained=model_config['use_pretrained'],
            hidden_dim=model_config['hidden_dim']
        )
    elif model_name == 'vit':
        model = ViTDeepFakeModel(
            model_name=model_config['vit_model'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            use_pretrained=model_config['use_pretrained'],
            hidden_dim=model_config['hidden_dim']
        )
    elif model_name == 'temporal':
        model = TemporalModel(
            feature_extractor='cnn',  # or 'vit'
            backbone=model_config['cnn_backbone'],
            temporal_model=model_config['temporal_model'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            hidden_dim=model_config['hidden_dim'],
            use_pretrained=model_config['use_pretrained']
        )
    elif model_name == 'ensemble':
        model = EnsembleModel(
            cnn_backbone=model_config['cnn_backbone'],
            vit_model=model_config['vit_model'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            hidden_dim=model_config['hidden_dim'],
            fusion_method='concat',
            use_pretrained=model_config['use_pretrained']
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train DeepFake Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Datasets
    train_dataset = DeepFakeDataset(
        data_dir=config['data']['train_dir'],
        config=config,
        is_training=True
    )
    val_dataset = DeepFakeDataset(
        data_dir=config['data']['val_dir'],
        config=config,
        is_training=False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Model
    model = build_model(config)
    print(f"Model: {config['model']['name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

