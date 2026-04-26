"""ML model for sensor-based fire risk prediction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Dict, Any, Union
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
        self.threshold = 0.5
        
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
        self.model.fit(X, y)
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
        features = np.array([[temperature, humidity]])
        probability = self.model.predict_proba(features)[0, 1]
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
        """Save trained model to disk."""
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'SensorFireRiskModel':
        """Load trained model from disk."""
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance