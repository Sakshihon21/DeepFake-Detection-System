"""Grad-CAM visualization for CNN models."""
import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class GradCAMExplainer:
    """Grad-CAM explainer for CNN models."""
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: Model to explain
            target_layer: Name of target layer (if None, uses last conv layer)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer
        if self.target_layer:
            target_module = dict(self.model.named_modules())[self.target_layer]
        else:
            # Find last convolutional layer
            target_module = None
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    target_module = module
        
        if target_module is None:
            raise ValueError("Could not find convolutional layer for Grad-CAM")
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Class index to visualize
            
        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        if output.dim() == 1:
            output[class_idx].backward()
        else:
            output[0, class_idx].backward()
        
        # Compute CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam
    
    def visualize(self, input_tensor: torch.Tensor, original_image: np.ndarray, 
                  class_idx: int = 1, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize Grad-CAM on image.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            original_image: Original image (BGR format)
            class_idx: Class index to visualize
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize original image if needed
        if original_image.shape[:2] != cam.shape:
            original_image = cv2.resize(original_image, (cam.shape[1], cam.shape[0]))
        
        # Overlay
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * original_rgb + 0.4 * heatmap).astype(np.uint8)
        
        # Save if path provided
        if save_path:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(original_rgb)
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return overlay

