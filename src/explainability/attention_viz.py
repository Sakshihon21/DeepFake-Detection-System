"""Attention visualization for Transformer models."""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import cv2


class AttentionVisualizer:
    """Visualize attention maps from Vision Transformer."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: ViT model
        """
        self.model = model
        self.model.eval()
        self.attention_weights = None
    
    def get_attention_weights(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights from model.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            
        Returns:
            Attention weights [num_layers, num_heads, num_patches, num_patches]
        """
        # Forward pass with attention output
        if hasattr(self.model, 'vit'):
            outputs = self.model.vit(pixel_values=input_tensor, output_attentions=True)
            attentions = outputs.attentions
        else:
            # If model doesn't have vit attribute, try to get attention directly
            raise NotImplementedError("Model structure not supported for attention visualization")
        
        return attentions
    
    def visualize_attention(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                           layer_idx: int = -1, head_idx: Optional[int] = None,
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize attention maps.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            original_image: Original image (BGR format)
            layer_idx: Which layer to visualize (-1 for last layer)
            head_idx: Which head to visualize (None for average)
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        attentions = self.get_attention_weights(input_tensor)
        
        # Get attention from specified layer
        layer_attention = attentions[layer_idx]  # [1, num_heads, num_patches+1, num_patches+1]
        layer_attention = layer_attention[0]  # [num_heads, num_patches+1, num_patches+1]
        
        # Average over heads or select specific head
        if head_idx is not None:
            attention_map = layer_attention[head_idx]
        else:
            attention_map = layer_attention.mean(dim=0)
        
        # Get attention to [CLS] token (first token)
        cls_attention = attention_map[0, 1:]  # Skip [CLS] token itself
        
        # Reshape to spatial dimensions
        # Assuming patch size 16 and image size 224
        patch_size = 16
        num_patches_per_side = int(np.sqrt(len(cls_attention)))
        attention_2d = cls_attention.reshape(num_patches_per_side, num_patches_per_side)
        
        # Resize to original image size
        h, w = original_image.shape[:2]
        attention_2d = cv2.resize(attention_2d, (w, h))
        
        # Normalize
        attention_2d = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attention_2d * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
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
            plt.imshow(attention_2d, cmap='hot')
            plt.title('Attention Map')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return overlay
    
    def visualize_multi_head(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                            layer_idx: int = -1, save_path: Optional[str] = None):
        """
        Visualize attention from all heads.
        
        Args:
            input_tensor: Input image tensor
            original_image: Original image
            layer_idx: Which layer to visualize
            save_path: Path to save visualization
        """
        attentions = self.get_attention_weights(input_tensor)
        layer_attention = attentions[layer_idx][0]  # [num_heads, num_patches+1, num_patches+1]
        num_heads = layer_attention.shape[0]
        
        # Get [CLS] attention for each head
        cls_attentions = layer_attention[:, 0, 1:]  # [num_heads, num_patches]
        
        # Reshape
        num_patches_per_side = int(np.sqrt(cls_attentions.shape[1]))
        
        # Create grid
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        plt.figure(figsize=(cols * 3, rows * 3))
        
        for head_idx in range(num_heads):
            plt.subplot(rows, cols, head_idx + 1)
            attention_2d = cls_attentions[head_idx].reshape(num_patches_per_side, num_patches_per_side)
            h, w = original_image.shape[:2]
            attention_2d = cv2.resize(attention_2d, (w, h))
            plt.imshow(attention_2d, cmap='hot')
            plt.title(f'Head {head_idx}')
            plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

