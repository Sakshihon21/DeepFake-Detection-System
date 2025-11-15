"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.metrics import roc_curve
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = 0.0
    
    # PR-AUC
    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['pr_auc'] = 0.0
    
    # EER (Equal Error Rate)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        metrics['eer'] = eer
    except:
        metrics['eer'] = 0.0
    
    # TPR@FPR
    try:
        fpr, tpr, _ = sk_roc_curve(y_true, y_prob)
        # TPR at FPR = 0.01, 0.05, 0.1
        for target_fpr in [0.01, 0.05, 0.1]:
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                metrics[f'tpr_at_fpr_{target_fpr}'] = tpr[idx[-1]]
            else:
                metrics[f'tpr_at_fpr_{target_fpr}'] = 0.0
    except:
        for target_fpr in [0.01, 0.05, 0.1]:
            metrics[f'tpr_at_fpr_{target_fpr}'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_robustness_metrics(
    y_true: np.ndarray,
    y_pred_original: np.ndarray,
    y_pred_perturbed: np.ndarray,
    perturbation_type: str
) -> Dict[str, float]:
    """
    Compute robustness metrics.
    
    Args:
        y_true: True labels
        y_pred_original: Predictions on original data
        y_pred_perturbed: Predictions on perturbed data
        perturbation_type: Type of perturbation
        
    Returns:
        Dictionary of robustness metrics
    """
    # Accuracy drop
    acc_original = accuracy_score(y_true, y_pred_original)
    acc_perturbed = accuracy_score(y_true, y_pred_perturbed)
    acc_drop = acc_original - acc_perturbed
    
    # Prediction consistency
    consistency = (y_pred_original == y_pred_perturbed).mean()
    
    metrics = {
        f'{perturbation_type}_acc_original': acc_original,
        f'{perturbation_type}_acc_perturbed': acc_perturbed,
        f'{perturbation_type}_acc_drop': acc_drop,
        f'{perturbation_type}_consistency': consistency
    }
    
    return metrics

