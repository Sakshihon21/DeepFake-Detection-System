"""Model evaluator."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
import json
from .metrics import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    compute_robustness_metrics
)
import cv2
import albumentations as A


class Evaluator:
    """Evaluator for deepfake detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Results directory
        results_dir = Path(config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir
    
    def evaluate(self, frame_aggregation: str = "majority") -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            frame_aggregation: How to aggregate frame predictions for videos
            
        Returns:
            Dictionary of metrics
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs if probs.ndim == 0 else probs.tolist())
                all_preds.extend(preds if preds.ndim == 0 else preds.tolist())
                all_labels.extend(target.cpu().numpy().tolist())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        
        # Plot curves
        plot_confusion_matrix(all_labels, all_preds, 
                            save_path=self.results_dir / 'confusion_matrix.png')
        plot_roc_curve(all_labels, all_probs, 
                      save_path=self.results_dir / 'roc_curve.png')
        plot_pr_curve(all_labels, all_probs, 
                     save_path=self.results_dir / 'pr_curve.png')
        
        # Save metrics
        metrics_path = self.results_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def evaluate_robustness(self) -> Dict[str, float]:
        """Evaluate model robustness to perturbations."""
        robustness_results = {}
        
        eval_config = self.config.get('evaluation', {})
        robustness_config = self.config.get('robustness', {})
        
        # Compression robustness
        if eval_config.get('robustness_tests', {}).get('compression', False):
            compression_metrics = self._test_compression(robustness_config.get('compression_qualities', [50, 70, 90]))
            robustness_results.update(compression_metrics)
        
        # Resolution robustness
        if eval_config.get('robustness_tests', {}).get('resolution', False):
            resolution_metrics = self._test_resolution(robustness_config.get('resolutions', [112, 160, 192]))
            robustness_results.update(resolution_metrics)
        
        # Blur robustness
        if eval_config.get('robustness_tests', {}).get('blur', False):
            blur_metrics = self._test_blur()
            robustness_results.update(blur_metrics)
        
        # Save results
        robustness_path = self.results_dir / 'robustness_metrics.json'
        with open(robustness_path, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        
        return robustness_results
    
    def _test_compression(self, qualities: List[int]) -> Dict[str, float]:
        """Test robustness to JPEG compression."""
        results = {}
        
        # Get original predictions
        original_preds, original_labels = self._get_predictions()
        
        for quality in qualities:
            # Apply compression and get predictions
            compressed_preds = self._get_predictions_with_compression(quality)
            
            # Compute metrics
            metrics = compute_robustness_metrics(
                original_labels, original_preds, compressed_preds, f'compression_{quality}'
            )
            results.update(metrics)
        
        return results
    
    def _test_resolution(self, resolutions: List[int]) -> Dict[str, float]:
        """Test robustness to resolution changes."""
        results = {}
        original_preds, original_labels = self._get_predictions()
        
        for res in resolutions:
            resized_preds = self._get_predictions_with_resolution(res)
            metrics = compute_robustness_metrics(
                original_labels, original_preds, resized_preds, f'resolution_{res}'
            )
            results.update(metrics)
        
        return results
    
    def _test_blur(self) -> Dict[str, float]:
        """Test robustness to blur."""
        original_preds, original_labels = self._get_predictions()
        blurred_preds = self._get_predictions_with_blur()
        
        metrics = compute_robustness_metrics(
            original_labels, original_preds, blurred_preds, 'blur'
        )
        return metrics
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions on original data."""
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds if preds.ndim == 0 else preds.tolist())
                all_labels.extend(target.numpy().tolist())
        
        return np.array(all_preds), np.array(all_labels)
    
    def _get_predictions_with_compression(self, quality: int) -> np.ndarray:
        """Get predictions with JPEG compression applied."""
        all_preds = []
        
        transform = A.Compose([
            A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                # Apply compression (simplified - would need to process each image)
                # This is a placeholder - full implementation would process each frame
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds if preds.ndim == 0 else preds.tolist())
        
        return np.array(all_preds)
    
    def _get_predictions_with_resolution(self, resolution: int) -> np.ndarray:
        """Get predictions with resolution change."""
        # Similar to compression - placeholder
        all_preds = []
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                # Resize would be applied in preprocessing
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds if preds.ndim == 0 else preds.tolist())
        
        return np.array(all_preds)
    
    def _get_predictions_with_blur(self) -> np.ndarray:
        """Get predictions with blur applied."""
        # Similar to compression - placeholder
        all_preds = []
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds if preds.ndim == 0 else preds.tolist())
        
        return np.array(all_preds)

