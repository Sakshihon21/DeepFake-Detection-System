"""Script to visualize model explanations."""
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

from src.utils.config import load_config
from src.models import CNNModel, ViTDeepFakeModel, EnsembleModel
from src.explainability import GradCAMExplainer, AttentionVisualizer
from src.utils.preprocessing import FrameProcessor


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
        raise ValueError(f"Model {model_name} not supported for visualization")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize Model Explanations')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='explanation.png')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Load model
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and process image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not load image: {args.image}")
    
    frame_processor = FrameProcessor(config, is_training=False)
    processed = frame_processor.process_frame(image, detect_face=True)
    
    if processed is None:
        processed = frame_processor.process_frame(image, detect_face=False)
    
    input_tensor = processed.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = "Fake" if prob > 0.5 else "Real"
    
    print(f"Prediction: {pred} (confidence: {prob:.4f})")
    
    # Visualize
    model_name = config['model']['name']
    output_path = Path(args.output)
    
    if model_name == 'cnn' or (model_name == 'ensemble'):
        # Grad-CAM for CNN
        explainer = GradCAMExplainer(model)
        vis = explainer.visualize(input_tensor, image, class_idx=1, save_path=str(output_path))
        print(f"Grad-CAM visualization saved to {output_path}")
    
    if model_name == 'vit' or model_name == 'ensemble':
        # Attention visualization for ViT
        if hasattr(model, 'vit') or model_name == 'vit':
            visualizer = AttentionVisualizer(model)
            vis = visualizer.visualize_attention(input_tensor, image, save_path=str(output_path))
            print(f"Attention visualization saved to {output_path}")


if __name__ == '__main__':
    main()

