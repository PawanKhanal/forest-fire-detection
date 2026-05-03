"""ML model for sensor-based fire risk prediction."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import joblib

class SensorFireRiskModel:
    """Machine learning model for fire risk based on temperature and humidity."""
    
    def __init__(self, model_type: str = 'random_forest') -> None:
        """
        Initialize sensor-based prediction model.
        
        Args:
            model_type: Type of model ('random_forest' or 'logistic')
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.is_fitted = False
        
    def _create_model(self):
        """Factory method to create the appropriate model."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        return models.get(self.model_type, models['random_forest'])
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'SensorFireRiskModel':
        """
        Train the model on historical sensor data.
        
        Args:
            X: Feature matrix [temperature, humidity]
            y: Target labels (0: no fire, 1: fire)
            
        Returns:
            Self for method chaining
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, temperature: float, humidity: float) -> Dict[str, Any]:
        """
        Predict fire risk from sensor readings.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        features = np.array([[temperature, humidity]])
        features_scaled = self.scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0, 1]
        prediction = int(probability >= self.threshold)
        
        return {
            'fire_risk': prediction,
            'probability': float(probability),
            'temperature': temperature,
            'humidity': humidity,
            'risk_level': self._get_risk_level(probability)
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Convert probability to risk level category.
        
        Args:
            probability: Fire probability (0-1)
            
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def save(self, filepath: str) -> None:
        """Save trained model and scaler to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'SensorFireRiskModel':
        """Load trained model from disk."""
        data = joblib.load(filepath)
        
        if isinstance(data, dict):
            instance = cls(model_type=data.get('model_type', 'random_forest'))
            instance.model = data['model']
            instance.scaler = data.get('scaler', StandardScaler())
            instance.is_fitted = data.get('is_fitted', True)
        else:
            instance = cls()
            instance.model = data
        
        return instance