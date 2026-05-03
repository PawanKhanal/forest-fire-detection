"""Unified prediction system for forest fire detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from models.cnn_model_pytorch import ForestFireCNN
from models.sensor_model import SensorFireRiskModel



@dataclass
class PredictionResult:
    """Data class for prediction results."""
    
    prediction: int
    confidence: float
    risk_level: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'metadata': self.metadata
        }


class PredictionInput(ABC):
    """Abstract base class for prediction input types."""
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the input."""
        pass
    
    @abstractmethod
    def get_input_type(self) -> str:
        """Get the type of input."""
        pass


class ImagePredictionInput(PredictionInput):
    """Input class for image-based fire detection."""
    
    def __init__(self, image_path: Union[str, Path], image_size: tuple = (224, 224)):
        """
        Initialize image input.
        
        Args:
            image_path: Path to the image file
            image_size: Target image size for model
        """
        self.image_path = Path(image_path)
        self.image_size = image_size
    
    def validate(self) -> bool:
        """Validate image input exists and is readable."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        if self.image_path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid image format: {self.image_path.suffix}")
        
        return True
    
    def get_input_type(self) -> str:
        """Get input type identifier."""
        return "image"
    
    def load_image(self) -> torch.Tensor:
        """Load and preprocess image for CNN model."""
        img = Image.open(self.image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transform(img).unsqueeze(0)


class SensorPredictionInput(PredictionInput):
    """Input class for sensor-based fire detection."""
    
    def __init__(self, temperature: float, humidity: float):
        """
        Initialize sensor input.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage (0-100)
        """
        self.temperature = temperature
        self.humidity = humidity
    
    def validate(self) -> bool:
        """Validate sensor input ranges."""
        if not isinstance(self.temperature, (int, float)):
            raise TypeError(f"Temperature must be numeric: {type(self.temperature)}")
        if not isinstance(self.humidity, (int, float)):
            raise TypeError(f"Humidity must be numeric: {type(self.humidity)}")
        
        if not (-50 <= self.temperature <= 60):
            raise ValueError(f"Temperature out of range: {self.temperature}°C")
        if not (0 <= self.humidity <= 100):
            raise ValueError(f"Humidity out of range: {self.humidity}%")
        
        return True
    
    def get_input_type(self) -> str:
        """Get input type identifier."""
        return "sensor"


class FirePredictionSystem:
    """Unified fire prediction system combining image and sensor models."""
    
    def __init__(
        self,
        cnn_model_path: Union[str, Path],
        sensor_model_path: Union[str, Path],
        device: Optional[str] = None,
        image_size: tuple = (224, 224)
    ):
        """
        Initialize prediction system with both models.
        
        Args:
            cnn_model_path: Path to trained CNN model weights
            sensor_model_path: Path to trained sensor model pickle
            device: Device to run CNN on ('cpu' or 'cuda')
            image_size: Target image size for CNN model
            
        Raises:
            FileNotFoundError: If model files not found
            RuntimeError: If model loading fails
        """
        self.cnn_model_path = Path(cnn_model_path)
        self.sensor_model_path = Path(sensor_model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        self._validate_model_paths()
        self.cnn_model = self._load_cnn_model()
        self.sensor_model = self._load_sensor_model()
        
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
    
    def _validate_model_paths(self) -> None:
        """Validate that model files exist."""
        if not self.cnn_model_path.exists():
            raise FileNotFoundError(f"CNN model not found: {self.cnn_model_path}")
        if not self.sensor_model_path.exists():
            raise FileNotFoundError(f"Sensor model not found: {self.sensor_model_path}")
    
    def _load_cnn_model(self) -> ForestFireCNN:
        """Load trained CNN model."""
        try:
            model = ForestFireCNN.load(str(self.cnn_model_path))
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load CNN model: {str(e)}")
    
    def _load_sensor_model(self) -> SensorFireRiskModel:
        """Load trained sensor model."""
        try:
            model = SensorFireRiskModel.load(str(self.sensor_model_path))
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load sensor model: {str(e)}")
    
    def predict_from_image(self, image_input: ImagePredictionInput) -> PredictionResult:
        """
        Predict fire presence from image.
        
        Args:
            image_input: ImagePredictionInput instance
            
        Returns:
            PredictionResult with image analysis
            
        Raises:
            ValueError: If input validation fails
        """
        if not image_input.validate():
            raise ValueError("Invalid image input")
        
        try:
            image_tensor = image_input.load_image().to(self.device)
            
            with torch.no_grad():
                logits = self.cnn_model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                
                confidence, predicted_class = torch.max(probabilities, 1)
                prediction = predicted_class.item()
                confidence = confidence.item()
            
            class_names = {0: "NO_FIRE", 1: "FIRE"}
            risk_level = self._get_risk_level_from_confidence(confidence, prediction)
            
            return PredictionResult(
                prediction=prediction,
                confidence=confidence,
                risk_level=risk_level,
                metadata={
                    'image_path': str(image_input.image_path),
                    'class_name': class_names[prediction],
                    'model_type': 'CNN',
                    'device': self.device
                }
            )
        except Exception as e:
            raise RuntimeError(f"Image prediction failed: {str(e)}")
    
    def predict_from_sensors(self, sensor_input: SensorPredictionInput) -> PredictionResult:
        """
        Predict fire risk from sensor readings.
        
        Args:
            sensor_input: SensorPredictionInput instance
            
        Returns:
            PredictionResult with sensor analysis
            
        Raises:
            ValueError: If input validation fails
        """
        if not sensor_input.validate():
            raise ValueError("Invalid sensor input")
        
        try:
            result = self.sensor_model.predict(
                sensor_input.temperature,
                sensor_input.humidity
            )
            
            return PredictionResult(
                prediction=result['fire_risk'],
                confidence=result['probability'],
                risk_level=result['risk_level'],
                metadata={
                    'temperature': sensor_input.temperature,
                    'humidity': sensor_input.humidity,
                    'model_type': 'RandomForest',
                    'threshold': self.sensor_model.threshold
                }
            )
        except Exception as e:
            raise RuntimeError(f"Sensor prediction failed: {str(e)}")
    
    def combined_prediction(
        self,
        image_input: ImagePredictionInput,
        sensor_input: SensorPredictionInput,
        image_weight: float = 0.6,
        sensor_weight: float = 0.4
    ) -> PredictionResult:
        """
        Ensemble prediction combining image and sensor models.
        
        Args:
            image_input: ImagePredictionInput instance
            sensor_input: SensorPredictionInput instance
            image_weight: Weight for CNN prediction (0-1)
            sensor_weight: Weight for sensor prediction (0-1)
            
        Returns:
            Weighted ensemble PredictionResult
            
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if not (0.99 <= image_weight + sensor_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {image_weight + sensor_weight}")
        
        image_result = self.predict_from_image(image_input)
        sensor_result = self.predict_from_sensors(sensor_input)
        
        weighted_confidence = (
            image_result.confidence * image_weight +
            sensor_result.confidence * sensor_weight
        )
        
        ensemble_prediction = 1 if weighted_confidence >= 0.5 else 0
        ensemble_risk = self._get_risk_level_from_confidence(
            weighted_confidence,
            ensemble_prediction
        )
        
        return PredictionResult(
            prediction=ensemble_prediction,
            confidence=weighted_confidence,
            risk_level=ensemble_risk,
            metadata={
                'ensemble_type': 'weighted_average',
                'image_weight': image_weight,
                'sensor_weight': sensor_weight,
                'image_confidence': image_result.confidence,
                'sensor_confidence': sensor_result.confidence,
                'image_result': image_result.to_dict(),
                'sensor_result': sensor_result.to_dict()
            }
        )
    
    @staticmethod
    def _get_risk_level_from_confidence(confidence: float, prediction: int) -> str:
        """
        Map confidence and prediction to risk level.
        
        Args:
            confidence: Model confidence (0-1)
            prediction: Class prediction (0: no fire, 1: fire)
            
        Returns:
            Risk level string
        """
        if prediction == 0:
            return "LOW"
        else:
            if confidence < 0.6:
                return "MEDIUM"
            elif confidence < 0.8:
                return "HIGH"
            else:
                return "CRITICAL"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'cnn_model': {
                'path': str(self.cnn_model_path),
                'device': self.device,
                'image_size': self.image_size,
                'total_params': sum(p.numel() for p in self.cnn_model.parameters())
            },
            'sensor_model': {
                'path': str(self.sensor_model_path),
                'model_type': self.sensor_model.model_type,
                'threshold': self.sensor_model.threshold
            }
        }
