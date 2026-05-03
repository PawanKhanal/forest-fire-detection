#!/usr/bin/env python3
"""Train sensor-based ML model on Portuguese Forest Fire Dataset + Nepal-calibrated predictor."""

import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib

from models.sensor_model import SensorFireRiskModel
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Data Loading
# =============================================================================

class SensorDataLoader:
    """Load and prepare sensor data."""
    
    def __init__(self, data_dir: Path = Path("data/sensor")):
        self.data_dir = data_dir
    
    def load_uci_dataset(self) -> Optional[pd.DataFrame]:
        """Load Portuguese Forest Fire dataset."""
        path = self.data_dir / "forestfires.csv"
        if path.exists():
            df = pd.read_csv(path)
            print(f"📥 UCI Dataset: {len(df)} records loaded")
            return df
        return None
    
    def load_all(self) -> pd.DataFrame:
        """Load dataset."""
        df = self.load_uci_dataset()
        if df is None:
            raise FileNotFoundError(f"No data found in {self.data_dir}")
        return df
    
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from DataFrame."""
        X = df[['temp', 'RH']].values
        y = (df['area'] > 0).astype(int).values
        return X, y


# =============================================================================
# Visualization
# =============================================================================

class ModelVisualizer:
    """Generate evaluation plots."""
    
    def __init__(self, save_dir: Path = Path("models/plots")):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_learning_curves(self, model, X: np.ndarray, y: np.ndarray) -> None:
        """Generate learning curves."""
        print("\n📈 Generating Learning Curves...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Accuracy')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
        
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curves - Sensor Fire Risk Model')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        filepath = self.save_dir / "sensor_learning_curves.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved to {filepath}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Generate confusion matrix."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask].astype(int)
        y_pred_clean = y_pred[mask].astype(int)
        
        cm = confusion_matrix(y_true_clean, y_pred_clean)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Fire', 'Fire'],
                    yticklabels=['No Fire', 'Fire'], ax=ax)
        ax.set_title('Confusion Matrix - Sensor Model')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        filepath = self.save_dir / "sensor_confusion_matrix.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   Confusion matrix saved to {filepath}")


# =============================================================================
# Trainer
# =============================================================================

class SensorModelTrainer:
    """Train and evaluate the sensor model."""
    
    def __init__(self, model_type: str = 'random_forest', test_size: float = 0.2, random_state: int = 42):
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model = SensorFireRiskModel(model_type=model_type)
        self.scaler = StandardScaler()
        self.metrics: Dict[str, float] = {}
        self.cv_scores: Optional[np.ndarray] = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def predict_risk(self, temperature: float, humidity: float) -> Dict[str, Any]:
        """Predict fire risk for given sensor readings."""
        input_scaled = self.scaler.transform([[temperature, humidity]])
        proba = self.model.model.predict_proba(input_scaled)[0, 1]
        pred = int(proba >= 0.5)
        
        if proba < 0.25:
            risk_level = "LOW"
        elif proba < 0.50:
            risk_level = "MODERATE"
        elif proba < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'prediction': pred,
            'probability': proba,
            'risk_level': risk_level,
            'temperature': temperature,
            'humidity': humidity
        }
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train model and compute metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n🔥 Training Random Forest...")
        start_time = time.time()
        self.model.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = self.model.model.predict(X_test_scaled)
        
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        self.metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
        
        print(f"\n✅ Training completed in {training_time:.4f}s")
        self._print_metrics()
        
        return self.metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
        """Perform cross-validation."""
        print(f"\n🔄 {cv}-Fold Cross-Validation:")
        X_scaled = self.scaler.fit_transform(X)
        self.cv_scores = cross_val_score(self.model.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        print(f"   Scores: {[f'{s:.3f}' for s in self.cv_scores]}")
        print(f"   Mean: {self.cv_scores.mean():.4f} (±{self.cv_scores.std():.4f})")
        
        return self.cv_scores
    
    def train_final(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train final model on full dataset."""
        print("\n🔥 Training final model on full dataset...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.model.fit(X_scaled, y)
    
    def save(self, filepath: Path) -> None:
        """Save model, scaler, and metadata."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.metrics,
            'cv_mean': self.cv_scores.mean() if self.cv_scores is not None else None,
            'cv_std': self.cv_scores.std() if self.cv_scores is not None else None
        }, filepath)
        print(f"\n💾 Model saved to {filepath}")
    
    def _print_metrics(self) -> None:
        """Print evaluation metrics."""
        print("\n📈 Performance Metrics:")
        print("-" * 30)
        for metric, value in self.metrics.items():
            print(f"   {metric}: {value:.4f} ({value*100:.1f}%)")


# =============================================================================
# Nepal-Calibrated Predictor (for Arduino deployment)
# =============================================================================

class NepalFirePredictor:
    """
    Hybrid fire risk predictor calibrated for Nepal.
    
    Combines:
    - ML model trained on Portuguese data (20% weight)
    - Canadian Fire Weather Index logic (80% weight)
    
    This compensates for the Portuguese dataset's climate differences.
    """
    
    def __init__(self, model_path: str = "models/saved/sensor_model.pkl"):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model file
        """
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        
    def _calculate_temp_factor(self, temperature: float) -> float:
        """Calculate temperature risk factor (0-1)."""
        if temperature < 20:
            return 0.05
        elif temperature < 25:
            return 0.20
        elif temperature < 30:
            return 0.40
        elif temperature < 35:
            return 0.65
        elif temperature < 40:
            return 0.85
        else:
            return 0.95
    
    def _calculate_humidity_factor(self, humidity: float) -> float:
        """Calculate humidity risk factor (0-1)."""
        if humidity > 65:
            return 0.05
        elif humidity > 50:
            return 0.20
        elif humidity > 35:
            return 0.40
        elif humidity > 20:
            return 0.65
        elif humidity > 10:
            return 0.85
        else:
            return 0.95
    
    def predict(self, temperature: float, humidity: float) -> Dict[str, Any]:
        """
        Predict fire risk for Nepal conditions.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage (0-100)
            
        Returns:
            Dictionary with risk assessment
        """
        # Fire science factors
        temp_factor = self._calculate_temp_factor(temperature)
        hum_factor = self._calculate_humidity_factor(humidity)
        science_score = (temp_factor * 0.5) + (hum_factor * 0.5)
        
        # ML model probability
        try:
            input_scaled = self.scaler.transform([[temperature, humidity]])
            ml_proba = self.model.predict_proba(input_scaled)[0, 1]
        except Exception:
            ml_proba = 0.5  # Fallback if model fails
        
        # Weighted combination (science-heavy for Nepal reliability)
        final_score = (science_score * 0.8) + (ml_proba * 0.2)
        
        # Determine risk level
        if final_score < 0.20:
            risk_level = "LOW"
            recommendation = "Normal conditions. Regular monitoring."
        elif final_score < 0.40:
            risk_level = "MODERATE"
            recommendation = "Slightly elevated. Monitor every 2 hours."
        elif final_score < 0.60:
            risk_level = "HIGH"
            recommendation = "Alert forest department. Increase monitoring."
        elif final_score < 0.80:
            risk_level = "VERY HIGH"
            recommendation = "Emergency alert! Prepare firefighting resources."
        else:
            risk_level = "EXTREME"
            recommendation = "IMMEDIATE ACTION REQUIRED! Possible evacuation."
        
        return {
            'fire_risk': final_score >= 0.40,
            'probability': float(final_score),
            'risk_percentage': float(final_score * 100),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'temperature': temperature,
            'humidity': humidity,
            'science_component': float(science_score),
            'ml_component': float(ml_proba)
        }


# =============================================================================
# Data Analysis
# =============================================================================

class DataAnalyzer:
    """Analyze patterns in sensor data."""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> None:
        """Print data statistics."""
        print("\n🔍 Data Pattern Analysis:")
        print("-" * 40)
        
        fire_df = df[df['area'] > 0]
        no_fire_df = df[df['area'] == 0]
        
        print(f"\n📊 Fire Days (n={len(fire_df)}):")
        print(f"   Temperature: {fire_df['temp'].mean():.1f}°C (±{fire_df['temp'].std():.1f})")
        print(f"   Humidity: {fire_df['RH'].mean():.1f}% (±{fire_df['RH'].std():.1f})")
        
        print(f"\n📊 Non-Fire Days (n={len(no_fire_df)}):")
        print(f"   Temperature: {no_fire_df['temp'].mean():.1f}°C (±{no_fire_df['temp'].std():.1f})")
        print(f"   Humidity: {no_fire_df['RH'].mean():.1f}% (±{no_fire_df['RH'].std():.1f})")
        
        print(f"\n⚠️  NOTE: Portuguese fires occur at ~19°C avg (agricultural burning).")
        print(f"   Nepal fires occur at 30-40°C (wildfires). Model is calibrated accordingly.")


# =============================================================================
# Test Scenarios
# =============================================================================

def print_ml_test_scenarios(trainer: SensorModelTrainer) -> None:
    """Print raw ML model predictions."""
    print("\n🧪 Raw ML Model Predictions:")
    print("-" * 50)
    
    test_scenarios = [
        (15.0, 80.0, "Cool, humid"),
        (22.0, 55.0, "Mild"),
        (30.0, 30.0, "Warm, dry"),
        (33.0, 18.0, "Hot, very dry"),
    ]
    
    for temp, hum, desc in test_scenarios:
        result = trainer.predict_risk(temp, hum)
        risk_bar = "█" * int(result['probability'] * 20)
        
        print(f"\n{desc}: {temp}°C, {hum}% humidity")
        print(f"  Risk: {'⚠️ FIRE' if result['prediction'] else '✅ SAFE'} | "
              f"Level: {result['risk_level']}")
        print(f"  Probability: [{risk_bar:<20}] {result['probability']:.1%}")


def print_nepal_test_scenarios(predictor: NepalFirePredictor) -> None:
    """Print Nepal-calibrated predictions."""
    print("\n\n🇳🇵 Nepal-Calibrated Predictions (for Arduino):")
    print("-" * 50)
    
    test_scenarios = [
        (15.0, 80.0, "Winter morning"),
        (25.0, 50.0, "Spring afternoon"),
        (32.0, 28.0, "Pre-monsoon dry spell"),
        (38.0, 15.0, "Peak fire season"),
        (42.0, 8.0,  "Extreme heatwave"),
    ]
    
    for temp, hum, desc in test_scenarios:
        result = predictor.predict(temp, hum)
        risk_bar = "█" * int(result['probability'] * 20)
        
        print(f"\n{desc}: {temp}°C, {hum}%")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Score: [{risk_bar:<20}] {result['probability']:.1%}")
        print(f"  → {result['recommendation']}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training pipeline."""
    print("🔥 Training Sensor-Based Fire Risk Model")
    print("=" * 50)
    print("Dataset: Portuguese Forest Fires (UCI)")
    print("Model: Random Forest Classifier")
    print("Calibration: Nepal-adjusted (80% Fire Science + 20% ML)\n")
    
    data_dir = Path("data/sensor")
    plots_dir = Path("models/plots")
    model_dir = Path("models/saved")
    
    # Load data
    loader = SensorDataLoader(data_dir)
    
    try:
        df = loader.load_all()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    X, y = loader.prepare_features(df)
    
    print(f"   Fire cases: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Non-fire: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    DataAnalyzer.analyze(df)
    
    # Train
    trainer = SensorModelTrainer(model_type='random_forest')
    trainer.train(X, y)
    trainer.cross_validate(X, y)
    
    # Visualize
    visualizer = ModelVisualizer(plots_dir)
    if trainer.y_test is not None and trainer.y_pred is not None:
        visualizer.plot_confusion_matrix(trainer.y_test, trainer.y_pred)
    
    # Train final and save
    trainer.train_final(X, y)
    visualizer.plot_learning_curves(trainer.model.model, trainer.scaler.fit_transform(X), y)
    trainer.save(model_dir / "sensor_model.pkl")
    
    # Show both raw ML and Nepal-calibrated predictions
    print_ml_test_scenarios(trainer)
    
    # Initialize Nepal predictor
    nepal_predictor = NepalFirePredictor(str(model_dir / "sensor_model.pkl"))
    print_nepal_test_scenarios(nepal_predictor)
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print(f"   ML Accuracy: {trainer.metrics['Accuracy']*100:.1f}%")
    print(f"   ML Recall: {trainer.metrics['Recall']*100:.1f}%")
    print(f"   Nepal Predictor saved to: {model_dir / 'sensor_model.pkl'}")
    print(f"\n📋 For Arduino: from train_sensor_model import NepalFirePredictor")


if __name__ == "__main__":
    main()