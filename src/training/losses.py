"""Loss functions for deepfake detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [B, 1] or [B]
            targets: Ground truth labels [B]
            
        Returns:
            Focal loss value
        """
        # Ensure inputs are 2D
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(1)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.unsqueeze(1), reduction='none')
        
        # Focal term
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss."""
    
    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class
            reduction: Reduction method
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Predicted logits [B, 1] or [B]
            targets: Ground truth labels [B]
            
        Returns:
            Weighted BCE loss value
        """
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(1)
        
        pos_weight_tensor = torch.tensor(self.pos_weight, device=inputs.device)
        return F.binary_cross_entropy_with_logits(
            inputs, 
            targets.unsqueeze(1), 
            pos_weight=pos_weight_tensor,
            reduction=self.reduction
        )

