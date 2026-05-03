"""Unified inference interface for both image and sensor models."""

import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from torchvision import transforms
from PIL import Image

from src.models.cnn_model_pytorch import ForestFireCNN
from src.models.sensor_model import SensorFireRiskModel


class FirePredictionSystem:
    """Integrated prediction system combining PyTorch CNN and sklearn sensor models."""
    
    def __init__(
        self,
        cnn_model_path: str = "models/saved/forest_fire_cnn_final.pth",
        sensor_model_path: str = "models/saved/sensor_model.pkl"
    ) -> None:
        """
        Initialize the prediction system.
        
        Args:
            cnn_model_path: Path to saved PyTorch CNN model (.pth)
            sensor_model_path: Path to saved sensor model (.pkl)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cnn_model = None
        self.sensor_data = None
        self.scaler = None
        
        # Load CNN model
        if Path(cnn_model_path).exists():
            self.cnn_model = ForestFireCNN.load(cnn_model_path, num_classes=2)
            self.cnn_model = self.cnn_model.to(self.device)
            self.cnn_model.eval()
        
        # Load sensor model (it's saved as a dictionary)
        if Path(sensor_model_path).exists():
            self.sensor_data = joblib.load(sensor_model_path)
            self.scaler = self.sensor_data.get('scaler')
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['No Fire', 'Fire']
    
    def predict_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Predict fire from single image using CNN.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction dictionary with confidence and class
        """
        if self.cnn_model is None:
            raise ValueError("CNN model not loaded")
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.cnn_model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        confidence = confidence.item()
        predicted_class = predicted.item()
        
        return {
            'fire_detected': bool(predicted_class),
            'confidence': confidence,
            'class_name': self.class_names[predicted_class],
            'all_probabilities': {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'model_type': 'CNN'
        }
    
    def predict_from_sensors(
        self,
        temperature: float,
        humidity: float
    ) -> Dict[str, Any]:
        """
        Predict fire risk from temperature and humidity using Random Forest.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage (0-100)
            
        Returns:
            Prediction dictionary with risk assessment
        """
        if self.sensor_data is None:
            raise ValueError("Sensor model not loaded")
        
        model = self.sensor_data['model']
        scaler = self.sensor_data['scaler']
        
        # Scale input
        input_scaled = scaler.transform([[temperature, humidity]])
        
        # Get prediction
        probability = model.predict_proba(input_scaled)[0, 1]
        prediction = model.predict(input_scaled)[0]
        
        return {
            'fire_risk': bool(prediction),
            'probability': float(probability),
            'risk_percentage': float(probability * 100),
            'temperature': temperature,
            'humidity': humidity,
            'risk_level': self._get_risk_level(probability),
            'model_type': 'Random Forest'
        }
    
    def combined_prediction(
        self,
        image_path: Optional[str] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        image_weight: float = 0.6,
        sensor_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Ensemble prediction combining both models.
        
        Args:
            image_path: Path to image file (optional)
            temperature: Temperature in Celsius (optional)
            humidity: Relative humidity percentage (optional)
            image_weight: Weight for image model (default 0.6)
            sensor_weight: Weight for sensor model (default 0.4)
            
        Returns:
            Combined prediction dictionary
        """
        result = {
            'final_decision': None,
            'combined_confidence': None,
            'risk_level': None,
            'image_analysis': None,
            'sensor_analysis': None
        }
        
        # Get individual predictions
        if image_path and self.cnn_model:
            result['image_analysis'] = self.predict_from_image(image_path)
        
        if temperature is not None and humidity is not None and self.sensor_data:
            result['sensor_analysis'] = self.predict_from_sensors(temperature, humidity)
        
        # Combine predictions
        if result['image_analysis'] and result['sensor_analysis']:
            # Both available - weighted average
            combined = (
                image_weight * result['image_analysis']['confidence'] +
                sensor_weight * result['sensor_analysis']['probability']
            )
            result['combined_confidence'] = combined
            result['final_decision'] = combined >= 0.5
            result['risk_level'] = self._get_risk_level(combined)
            
        elif result['image_analysis']:
            # Only image
            result['combined_confidence'] = result['image_analysis']['confidence']
            result['final_decision'] = result['image_analysis']['fire_detected']
            result['risk_level'] = self._get_risk_level(result['combined_confidence'])
            
        elif result['sensor_analysis']:
            # Only sensor
            result['combined_confidence'] = result['sensor_analysis']['probability']
            result['final_decision'] = result['sensor_analysis']['fire_risk']
            result['risk_level'] = result['sensor_analysis']['risk_level']
        
        return result
    
    @staticmethod
    def _get_risk_level(confidence: float) -> str:
        """Determine risk level from confidence score."""
        if confidence < 0.25:
            return "LOW"
        elif confidence < 0.50:
            return "MODERATE"
        elif confidence < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return self.cnn_model is not None and self.sensor_data is not None


def main():
    """Test the unified prediction system."""
    
    print("🧪 Testing Fire Prediction System")
    print("=" * 60)
    
    # Initialize system
    system = FirePredictionSystem()
    
    print(f"\n📊 System Status:")
    print(f"   CNN Model: {'✅ Loaded' if system.cnn_model else '❌ Not found'}")
    print(f"   Sensor Model: {'✅ Loaded' if system.sensor_data else '❌ Not found'}")
    print(f"   Ready: {'✅ YES' if system.is_ready() else '❌ NO'}")
    
    # Test sensor prediction
    if system.sensor_data:
        print("\n🌡️  Sensor Prediction Tests:")
        print("-" * 40)
        
        test_cases = [
            (22.0, 70.0, "Mild morning"),
            (35.0, 25.0, "Hot afternoon"),
            (40.0, 12.0, "Extreme conditions"),
        ]
        
        for temp, hum, desc in test_cases:
            result = system.predict_from_sensors(temp, hum)
            print(f"\n{desc}:")
            print(f"   Temp: {temp}°C, Humidity: {hum}%")
            print(f"   Risk: {result['risk_level']} ({result['risk_percentage']:.1f}%)")
    
    # Test image prediction
    if system.cnn_model:
        test_image = Path("data/raw/test/fire")
        if test_image.exists():
            images = list(test_image.glob("*.jpg"))[:1]
            if images:
                print(f"\n🖼️  Image Prediction Test:")
                print("-" * 40)
                result = system.predict_from_image(str(images[0]))
                print(f"   Image: {images[0].name}")
                print(f"   Prediction: {result['class_name']}")
                print(f"   Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()