"""Unified inference interface for both image and sensor models."""

import numpy as np
from typing import Dict, Any, Union
from pathlib import Path
import tensorflow as tf

from src.models.cnn_model import ForestFireCNN
from src.models.sensor_model import SensorFireRiskModel

class FirePredictionSystem:
    """Integrated prediction system combining image and sensor models."""
    
    def __init__(
        self,
        cnn_model_path: str,
        sensor_model_path: str
    ) -> None:
        """
        Initialize the prediction system.
        
        Args:
            cnn_model_path: Path to saved CNN model
            sensor_model_path: Path to saved sensor model
        """
        self.cnn_model = ForestFireCNN.load(cnn_model_path)
        self.sensor_model = SensorFireRiskModel.load(sensor_model_path)
        
    def predict_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Predict fire from single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction dictionary
        """
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.cnn_model.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        probability = float(self.cnn_model.predict(img_array)[0, 0])
        prediction = int(probability >= 0.5)
        
        return {
            'fire_detected': bool(prediction),
            'confidence': probability,
            'model_type': 'CNN'
        }
    
    def predict_from_sensors(
        self,
        temperature: float,
        humidity: float
    ) -> Dict[str, Any]:
        """
        Predict fire risk from sensor readings.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage
            
        Returns:
            Prediction dictionary
        """
        return self.sensor_model.predict(temperature, humidity)
    
    def combined_prediction(
        self,
        image_path: str,
        temperature: float,
        humidity: float
    ) -> Dict[str, Any]:
        """
        Ensemble prediction combining both models.
        
        Args:
            image_path: Path to image file
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage
            
        Returns:
            Combined prediction with weighted average
        """
        image_pred = self.predict_from_image(image_path)
        sensor_pred = self.predict_from_sensors(temperature, humidity)
        
        combined_confidence = (
            0.6 * image_pred['confidence'] + 
            0.4 * sensor_pred['probability']
        )
        
        return {
            'image_analysis': image_pred,
            'sensor_analysis': sensor_pred,
            'combined_risk': combined_confidence,
            'final_decision': combined_confidence >= 0.5,
            'risk_level': self._calculate_risk_level(combined_confidence)
        }
    
    @staticmethod
    def _calculate_risk_level(confidence: float) -> str:
        """Determine risk level from confidence score."""
        if confidence < 0.3:
            return "LOW"
        elif confidence < 0.5:
            return "MODERATE"
        elif confidence < 0.7:
            return "HIGH"
        else:
            return "CRITICAL"