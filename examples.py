"""Example usage of the unified prediction system."""

from src.inference.predictor import (
    FirePredictionSystem,
    ImagePredictionInput,
    SensorPredictionInput
)


def example_image_prediction():
    """Example: Predict fire from image."""
    # Initialize system
    prediction_system = FirePredictionSystem(
        cnn_model_path='models/saved/forest_fire_cnn_final.pth',
        sensor_model_path='models/saved/sensor_model.pkl'
    )
    
    # Image-based prediction
    image_input = ImagePredictionInput(image_path='data/raw/test/fire/sample.jpg')
    result = prediction_system.predict_from_image(image_input)
    
    print(f"Image Prediction:")
    print(f"  Class: {result.metadata['class_name']}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Risk Level: {result.risk_level}")


def example_sensor_prediction():
    """Example: Predict fire risk from sensors."""
    prediction_system = FirePredictionSystem(
        cnn_model_path='models/saved/forest_fire_cnn_final.pth',
        sensor_model_path='models/saved/sensor_model.pkl'
    )
    
    # Sensor-based prediction
    sensor_input = SensorPredictionInput(temperature=35.5, humidity=25.0)
    result = prediction_system.predict_from_sensors(sensor_input)
    
    print(f"Sensor Prediction:")
    print(f"  Temperature: {sensor_input.temperature}°C")
    print(f"  Humidity: {sensor_input.humidity}%")
    print(f"  Fire Risk: {result.prediction}")
    print(f"  Probability: {result.confidence:.2%}")
    print(f"  Risk Level: {result.risk_level}")


def example_ensemble_prediction():
    """Example: Combined image and sensor prediction."""
    prediction_system = FirePredictionSystem(
        cnn_model_path='models/saved/forest_fire_cnn_final.pth',
        sensor_model_path='models/saved/sensor_model.pkl'
    )
    
    # Ensemble prediction
    image_input = ImagePredictionInput(image_path='data/raw/test/fire/sample.jpg')
    sensor_input = SensorPredictionInput(temperature=35.5, humidity=25.0)
    
    result = prediction_system.combined_prediction(
        image_input=image_input,
        sensor_input=sensor_input,
        image_weight=0.6,
        sensor_weight=0.4
    )
    
    print(f"Ensemble Prediction:")
    print(f"  Combined Risk: {result.prediction}")
    print(f"  Weighted Confidence: {result.confidence:.2%}")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Image Confidence: {result.metadata['image_confidence']:.2%}")
    print(f"  Sensor Confidence: {result.metadata['sensor_confidence']:.2%}")


def example_arduino_monitoring():
    """Example: Real-time monitoring from Arduino."""
    from arduino_reader import ArduinoDataSource, FireRiskMonitor
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize data source and monitor
    arduino_source = ArduinoDataSource(
        port=os.getenv('ARDUINO_PORT', '/dev/ttyUSB0'),
        baudrate=int(os.getenv('ARDUINO_BAUDRATE', '9600'))
    )
    
    monitor = FireRiskMonitor(
        data_source=arduino_source,
        sensor_model_path=os.getenv('SENSOR_MODEL_PATH', 'models/saved/sensor_model.pkl'),
        update_interval=5.0
    )
    
    # Start monitoring (runs until KeyboardInterrupt)
    monitor.start()
    
    # Get statistics
    stats = monitor.get_statistics()
    print(f"\nMonitoring Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    print("Forest Fire Detection System - Usage Examples\n")
    print("="*70 + "\n")
    
    print("1. Image Prediction Example")
    print("-"*70)
    try:
        example_image_prediction()
    except Exception as e:
        print(f"   Note: {str(e)}")
    
    print("\n2. Sensor Prediction Example")
    print("-"*70)
    try:
        example_sensor_prediction()
    except Exception as e:
        print(f"   Note: {str(e)}")
    
    print("\n3. Ensemble Prediction Example")
    print("-"*70)
    try:
        example_ensemble_prediction()
    except Exception as e:
        print(f"   Note: {str(e)}")
    
    print("\n4. Arduino Monitoring Example")
    print("-"*70)
    print("   Uncomment and run example_arduino_monitoring() to start monitoring")
    print("   (requires Arduino connected on ARDUINO_PORT)")
