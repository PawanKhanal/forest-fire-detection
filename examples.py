#!/usr/bin/env python3
"""Example usage of the forest fire detection system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.predictor import (
    FirePredictionSystem,
    ImagePredictionInput,
    SensorPredictionInput
)

CNN_MODEL_PATH = "models/saved/forest_fire_cnn_final.pth"
SENSOR_MODEL_PATH = "models/saved/sensor_model.pkl"


def example_system_status():
    """Check system status."""
    try:
        system = FirePredictionSystem(
            cnn_model_path=CNN_MODEL_PATH,
            sensor_model_path=SENSOR_MODEL_PATH
        )
        
        info = system.get_model_info()
        
        print("📊 System Status:")
        print("-" * 50)
        print(f"   CNN Model: ✅ Loaded ({info['cnn_model']['total_params']:,} params)")
        print(f"   Device: {info['cnn_model']['device']}")
        print(f"   Sensor Model: ✅ Loaded ({info['sensor_model']['model_type']})")
        print(f"   Ready: ✅ YES")
        
        return system
    except FileNotFoundError as e:
        print(f"   ❌ Model not found: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def example_sensor_prediction(system):
    """Example: Predict fire risk from sensors."""
    if system is None:
        return

    print("🌡️  Sensor Prediction Examples:")
    print("-" * 50)

    test_cases = [
        (22.0, 70.0, "Mild morning"),
        (30.0, 45.0, "Warm afternoon"),
        (36.0, 22.0, "Hot and dry"),
        (42.0, 10.0, "Extreme conditions"),
    ]

    for temp, hum, desc in test_cases:
        try:
            sensor_input = SensorPredictionInput(temperature=temp, humidity=hum)
            result = system.predict_from_sensors(sensor_input)

            risk_bar = "█" * int(result.confidence * 20)

            print(f"\n{desc}:")
            print(f"   Temperature: {temp}°C, Humidity: {hum}%")
            print(f"   Risk: {'⚠️ FIRE' if result.prediction else '✅ SAFE'}")
            print(f"   Level: {result.risk_level}")
            print(f"   Confidence: [{risk_bar:<20}] {result.confidence:.1%}")
        except Exception as e:
            print(f"\n{desc}: ❌ {e}")


def example_image_prediction(system):
    """Example: Predict fire from image."""
    if system is None:
        return

    # Find a test image
    images = []
    for split in ["test", "val"]:
        for cls in ["fire", "nofire"]:
            d = Path(f"data/raw/{split}/{cls}")
            if d.exists():
                found = list(d.glob("*.jpg"))[:1]
                images.extend(found)

    if not images:
        print("   ❌ No test images found")
        return

    print("🖼️  Image Prediction Examples:")
    print("-" * 50)

    for img_path in images[:2]:
        try:
            image_input = ImagePredictionInput(image_path=str(img_path))
            result = system.predict_from_image(image_input)

            print(f"\nImage: {img_path.name}")
            print(f"   Prediction: {result.metadata['class_name']}")
            print(f"   Confidence: {result.confidence:.2%}")
            print(f"   Risk Level: {result.risk_level}")
        except Exception as e:
            print(f"\n{img_path.name}: ❌ {e}")


def example_combined_prediction(system):
    """Example: Combined prediction."""
    if system is None:
        return

    images = []
    for cls in ["fire", "nofire"]:
        d = Path(f"data/raw/test/{cls}")
        if d.exists():
            images = list(d.glob("*.jpg"))[:1]
            if images:
                break

    if not images:
        print("   ❌ No test images found")
        return

    print("🔗 Combined Prediction Example:")
    print("-" * 50)

    try:
        image_input = ImagePredictionInput(image_path=str(images[0]))
        sensor_input = SensorPredictionInput(temperature=36.0, humidity=22.0)

        result = system.combined_prediction(
            image_input=image_input,
            sensor_input=sensor_input,
            image_weight=0.6,
            sensor_weight=0.4
        )

        print(f"\nImage: {images[0].name}")
        print(f"Sensors: 36.0°C, 22.0% humidity")
        print(f"\nResults:")
        print(f"   Combined Risk: {result.risk_level}")
        print(f"   Weighted Confidence: {result.confidence:.2%}")
        print(f"   Image Confidence: {result.metadata['image_confidence']:.2%}")
        print(f"   Sensor Confidence: {result.metadata['sensor_confidence']:.2%}")
    except Exception as e:
        print(f"   ❌ {e}")


def main():
    """Run all examples."""
    print("🌲 Forest Fire Detection System - Usage Examples")
    print("=" * 60)
    print()

    system = example_system_status()

    print("\n" + "=" * 60)
    print()

    example_sensor_prediction(system)

    print("\n" + "=" * 60)
    print()

    example_image_prediction(system)

    print("\n" + "=" * 60)
    print()

    example_combined_prediction(system)

    print("\n" + "=" * 60)
    print("✅ Examples complete!")


if __name__ == "__main__":
    main()