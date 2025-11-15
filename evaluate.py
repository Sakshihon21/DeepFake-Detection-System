"""Main evaluation script."""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils.config import load_config
from src.data.dataset import DeepFakeDataset
from src.models import CNNModel, ViTDeepFakeModel, TemporalModel, EnsembleModel
from src.evaluation import Evaluator


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
            feature_extractor='cnn',
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
    parser = argparse.ArgumentParser(description='Evaluate DeepFake Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default=None, help='Override test directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override test directory if provided
    if args.test_dir:
        config['data']['test_dir'] = args.test_dir
    
    # Device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Test dataset
    test_dataset = DeepFakeDataset(
        data_dir=config['data']['test_dir'],
        config=config,
        is_training=False
    )
    
    # Data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Model
    model = build_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluator.evaluate()
    
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Robustness evaluation
    if config.get('evaluation', {}).get('robustness_tests', {}):
        print("\nEvaluating robustness...")
        robustness_metrics = evaluator.evaluate_robustness()
        print("\n=== Robustness Results ===")
        for key, value in robustness_metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()

