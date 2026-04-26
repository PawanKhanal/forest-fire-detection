"""Quick start script for testing the model without full dataset."""

import numpy as np
from src.models.sensor_model import SensorFireRiskModel

def create_sample_sensor_data():
    """Create synthetic sensor data for initial testing."""
    np.random.seed(42)
    n_samples = 1000
    
    temperatures = np.random.uniform(20, 45, n_samples)
    humidities = np.random.uniform(20, 80, n_samples)
    
    fire_risk = (
        (temperatures > 35) & 
        (humidities < 35)
    ).astype(int)
    
    X = np.column_stack([temperatures, humidities])
    y = fire_risk
    
    return X, y

def main():
    """Quick test of sensor model."""
    print("Creating synthetic sensor data...")
    X, y = create_sample_sensor_data()
    
    print("Training sensor model...")
    model = SensorFireRiskModel(model_type='random_forest')
    model.train(X, y)
    
    test_cases = [
        (25.0, 60.0, "Normal day"),
        (38.0, 25.0, "Hot and dry - High risk"),
        (42.0, 15.0, "Extreme conditions - Critical risk")
    ]
    
    print("\nTesting predictions:")
    print("-" * 50)
    
    for temp, hum, desc in test_cases:
        result = model.predict(temp, hum)
        print(f"\nScenario: {desc}")
        print(f"Temperature: {temp}°C, Humidity: {hum}%")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability: {result['probability']:.3f}")

if __name__ == "__main__":
    main()