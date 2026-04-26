"""Evaluation metrics and model performance utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self) -> None:
        """Initialize evaluator."""
        self.metrics = {}
        
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary of metric values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = ['No Fire', 'Fire'],
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate and display confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix - Forest Fire Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()