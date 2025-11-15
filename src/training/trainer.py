"""Training utilities."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Optional
from ..training.losses import FocalLoss, WeightedBCELoss


class Trainer:
    """Trainer class for deepfake detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        train_config = config['training']
        loss_type = train_config.get('loss', 'bce')
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=train_config.get('focal_alpha', 0.25),
                gamma=train_config.get('focal_gamma', 2.0)
            )
        elif loss_type == 'weighted_bce':
            # Calculate pos_weight from data if needed
            pos_weight = train_config.get('pos_weight', 1.0)
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        optimizer_type = train_config.get('optimizer', 'adamw')
        lr = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        
        # Scheduler
        scheduler_type = train_config.get('scheduler', 'cosine')
        num_epochs = train_config.get('num_epochs', 50)
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=num_epochs // 3, gamma=0.1)
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
        # Mixed precision
        self.use_amp = train_config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Paths
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), target)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output.squeeze(), target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, num_epochs: Optional[int] = None):
        """Train model for multiple epochs."""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        patience = self.config['training'].get('early_stopping_patience', 10)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Save best model
            save_best = self.config['training'].get('save_best_only', True)
            if save_best:
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint(epoch, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}, val_acc: {self.best_val_acc:.4f}")

