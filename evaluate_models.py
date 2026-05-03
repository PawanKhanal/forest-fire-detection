"""Model evaluation and comparison script."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score
)
import torch
from torchvision import datasets, transforms
import joblib

from models.cnn_model_pytorch import ForestFireCNN
from models.sensor_model import SensorFireRiskModel
from src.data.pytorch_dataset import ForestFireDataLoader

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """Evaluate and compare forest fire detection models."""
    
    def __init__(
        self,
        cnn_model_path: str,
        sensor_model_path: str,
        output_dir: str = 'models/plots'
    ):
        """
        Initialize model evaluator.
        
        Args:
            cnn_model_path: Path to trained CNN model
            sensor_model_path: Path to trained sensor model
            output_dir: Directory to save evaluation plots
        """
        self.cnn_model_path = cnn_model_path
        self.sensor_model_path = sensor_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_model = None
        self.sensor_model = None
    
    def load_models(self) -> None:
        """Load trained models."""
        try:
            print("Loading CNN model...")
            self.cnn_model = ForestFireCNN.load(self.cnn_model_path)
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            print(f"✓ CNN model loaded from {self.cnn_model_path}")
            
            print("Loading sensor model...")
            self.sensor_model = SensorFireRiskModel.load(self.sensor_model_path)
            print(f"✓ Sensor model loaded from {self.sensor_model_path}")
        except Exception as e:
            print(f"✗ Error loading models: {str(e)}")
            raise
    
    def evaluate_cnn_model(
        self,
        test_data_dir: str = 'data/raw/test',
        image_size: Tuple[int, int] = (224, 224)
    ) -> Dict[str, Any]:
        """
        Evaluate CNN model on test dataset.
        
        Args:
            test_data_dir: Path to test images organized in class folders
            image_size: Image size for model input
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*70)
        print("CNN IMAGE MODEL EVALUATION")
        print("="*70)
        
        # Load test data
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        try:
            test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False
            )
            print(f"✓ Loaded {len(test_dataset)} test images from {test_data_dir}")
        except Exception as e:
            print(f"✗ Error loading test data: {str(e)}")
            return {}
        
        # Run inference
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.cnn_model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(
            all_targets,
            all_predictions,
            target_names=['NO_FIRE', 'FIRE']
        ))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def evaluate_sensor_model(
    self,
    test_csv: str = 'data/sensor/forestfires.csv'
) -> Dict[str, Any]:
        """
        Evaluate sensor model on Portuguese Forest Fire data.
        """
        print("\n" + "="*70)
        print("SENSOR MODEL EVALUATION")
        print("="*70)
        
        import pandas as pd
        
        # Load data
        df = pd.read_csv(test_csv)
        print(f"✓ Loaded sensor data from {test_csv}")
        print(f"  Total samples: {len(df)}")
        
        # Extract features and labels
        X = df[['temp', 'RH']].values
        y = (df['area'] > 0).astype(int).values  # Binary: area > 0 = fire
        
        print(f"  Fire cases: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Non-fire: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
        
        # Load model and scaler from saved dict
        data = joblib.load(self.sensor_model_path)
        model = data['model']
        scaler = data['scaler']
        
        # Split data for honest evaluation (same split as training)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale test features
        X_scaled = scaler.transform(X_test)
        
        # Predict on held-out test set
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Use test labels for evaluation
        y = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(
            y,
            predictions,
            target_names=['NO_FIRE', 'FIRE'],
            zero_division=0
        ))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'targets': y,
            'probabilities': probabilities
        }
    
    def plot_confusion_matrices(
        self,
        cnn_metrics: Dict[str, Any],
        sensor_metrics: Dict[str, Any]
    ) -> None:
        """Plot confusion matrices for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # CNN confusion matrix
        if 'targets' in cnn_metrics and 'predictions' in cnn_metrics:
            cm_cnn = confusion_matrix(
                cnn_metrics['targets'],
                cnn_metrics['predictions']
            )
            sns.heatmap(
                cm_cnn,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[0],
                cbar=False,
                xticklabels=['NO_FIRE', 'FIRE'],
                yticklabels=['NO_FIRE', 'FIRE']
            )
            axes[0].set_title('CNN Model - Confusion Matrix', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
        
        # Sensor confusion matrix
        if 'targets' in sensor_metrics and 'predictions' in sensor_metrics:
            cm_sensor = confusion_matrix(
                sensor_metrics['targets'],
                sensor_metrics['predictions']
            )
            sns.heatmap(
                cm_sensor,
                annot=True,
                fmt='d',
                cmap='Oranges',
                ax=axes[1],
                cbar=False,
                xticklabels=['NO_FIRE', 'FIRE'],
                yticklabels=['NO_FIRE', 'FIRE']
            )
            axes[1].set_title('Sensor Model - Confusion Matrix', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        filepath = self.output_dir / 'confusion_matrices.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrices to {filepath}")
        plt.close()
    
    def plot_roc_curves(
        self,
        cnn_metrics: Dict[str, Any],
        sensor_metrics: Dict[str, Any]
    ) -> None:
        """Plot ROC curves for both models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # CNN ROC curve
        if 'targets' in cnn_metrics and 'probabilities' in cnn_metrics:
            fpr_cnn, tpr_cnn, _ = roc_curve(
                cnn_metrics['targets'],
                cnn_metrics['probabilities']
            )
            roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
            ax.plot(
                fpr_cnn, tpr_cnn,
                label=f'CNN (AUC = {roc_auc_cnn:.3f})',
                linewidth=2.5,
                color='#1f77b4'
            )
        
        # Sensor ROC curve
        if 'targets' in sensor_metrics and 'probabilities' in sensor_metrics:
            fpr_sensor, tpr_sensor, _ = roc_curve(
                sensor_metrics['targets'],
                sensor_metrics['probabilities']
            )
            roc_auc_sensor = auc(fpr_sensor, tpr_sensor)
            ax.plot(
                fpr_sensor, tpr_sensor,
                label=f'Sensor Model (AUC = {roc_auc_sensor:.3f})',
                linewidth=2.5,
                color='#ff7f0e'
            )
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'roc_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves to {filepath}")
        plt.close()
    
    def plot_precision_recall_curves(
        self,
        cnn_metrics: Dict[str, Any],
        sensor_metrics: Dict[str, Any]
    ) -> None:
        """Plot precision-recall curves for both models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # CNN PR curve
        if 'targets' in cnn_metrics and 'probabilities' in cnn_metrics:
            precision_cnn, recall_cnn, _ = precision_recall_curve(
                cnn_metrics['targets'],
                cnn_metrics['probabilities']
            )
            ax.plot(
                recall_cnn, precision_cnn,
                label='CNN Model',
                linewidth=2.5,
                color='#1f77b4',
                marker='o',
                markersize=3
            )
        
        # Sensor PR curve
        if 'targets' in sensor_metrics and 'probabilities' in sensor_metrics:
            precision_sensor, recall_sensor, _ = precision_recall_curve(
                sensor_metrics['targets'],
                sensor_metrics['probabilities']
            )
            ax.plot(
                recall_sensor, precision_sensor,
                label='Sensor Model',
                linewidth=2.5,
                color='#ff7f0e',
                marker='s',
                markersize=3
            )
        
        ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax.set_title('Precision-Recall Curves Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        filepath = self.output_dir / 'precision_recall_curves.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved precision-recall curves to {filepath}")
        plt.close()
    
    def plot_metrics_comparison(
        self,
        cnn_metrics: Dict[str, Any],
        sensor_metrics: Dict[str, Any]
    ) -> None:
        """Plot comparison bar chart of model metrics."""
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        cnn_values = [
            cnn_metrics.get('accuracy', 0),
            cnn_metrics.get('precision', 0),
            cnn_metrics.get('recall', 0),
            cnn_metrics.get('f1_score', 0)
        ]
        sensor_values = [
            sensor_metrics.get('accuracy', 0),
            sensor_metrics.get('precision', 0),
            sensor_metrics.get('recall', 0),
            sensor_metrics.get('f1_score', 0)
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(
            x - width/2, cnn_values,
            width,
            label='CNN Model',
            color='#1f77b4',
            alpha=0.8
        )
        bars2 = ax.bar(
            x + width/2, sensor_values,
            width,
            label='Sensor Model',
            color='#ff7f0e',
            alpha=0.8
        )
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9
                )
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        filepath = self.output_dir / 'metrics_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics comparison to {filepath}")
        plt.close()
    
    def generate_report(
        self,
        cnn_metrics: Dict[str, Any],
        sensor_metrics: Dict[str, Any]
    ) -> None:
        """Generate and save evaluation report."""
        report = f"""
FOREST FIRE DETECTION SYSTEM - MODEL EVALUATION REPORT
{'='*70}

Generated: {Path('').resolve()}
Output Directory: {self.output_dir}

CNN IMAGE MODEL
{'-'*70}
Accuracy:  {cnn_metrics.get('accuracy', 0):.4f}
Precision: {cnn_metrics.get('precision', 0):.4f}
Recall:    {cnn_metrics.get('recall', 0):.4f}
F1-Score:  {cnn_metrics.get('f1_score', 0):.4f}

SENSOR MODEL (Random Forest)
{'-'*70}
Accuracy:  {sensor_metrics.get('accuracy', 0):.4f}
Precision: {sensor_metrics.get('precision', 0):.4f}
Recall:    {sensor_metrics.get('recall', 0):.4f}
F1-Score:  {sensor_metrics.get('f1_score', 0):.4f}

RECOMMENDATIONS
{'-'*70}
1. CNN Model: Best for image-based fire detection with pixel-level features
2. Sensor Model: Effective for rapid assessment using weather conditions
3. Ensemble: Combine both models for robust predictions

Generated Plots:
  - confusion_matrices.png
  - roc_curves.png
  - precision_recall_curves.png
  - metrics_comparison.png
"""
        
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Evaluation report saved to {report_path}")
        print(report)
    
    def evaluate_all(self) -> None:
        """Run complete evaluation pipeline."""
        print("\n" + "="*70)
        print("FOREST FIRE DETECTION - COMPLETE MODEL EVALUATION")
        print("="*70 + "\n")
        
        self.load_models()
        
        cnn_metrics = self.evaluate_cnn_model()
        sensor_metrics = self.evaluate_sensor_model()
        
        print("\n" + "="*70)
        print("GENERATING EVALUATION PLOTS")
        print("="*70 + "\n")
        
        if cnn_metrics and sensor_metrics:
            self.plot_confusion_matrices(cnn_metrics, sensor_metrics)
            self.plot_roc_curves(cnn_metrics, sensor_metrics)
            self.plot_precision_recall_curves(cnn_metrics, sensor_metrics)
            self.plot_metrics_comparison(cnn_metrics, sensor_metrics)
            self.generate_report(cnn_metrics, sensor_metrics)
        else:
            print("✗ Insufficient metrics for plotting")
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"All plots saved to: {self.output_dir}")


def main() -> None:
    """Main entry point."""
    cnn_model_path = os.getenv(
        'CNN_MODEL_PATH',
        'models/saved/forest_fire_cnn_final.pth'
    )
    sensor_model_path = os.getenv(
        'SENSOR_MODEL_PATH',
        'models/saved/sensor_model.pkl'
    )
    
    evaluator = ModelEvaluator(cnn_model_path, sensor_model_path)
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()
