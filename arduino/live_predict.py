#!/usr/bin/env python3
"""Live prediction from Arduino DHT11 sensor."""

import serial
import time
import sys
from pathlib import Path

import joblib
import numpy as np


def get_risk_level(probability):
    if probability < 0.35:
        return "LOW", "✅ Safe"
    elif probability < 0.50:
        return "MODERATE", "⚠️ Caution"
    elif probability < 0.65:
        return "HIGH", "⚠️ Warning"
    else:
        return "CRITICAL", "🔥 FIRE RISK"


def main():
    model_path = Path("models/saved/sensor_model.pkl")
    if not model_path.exists():
        print("❌ Model not found!")
        print("   Run: python train_sensor_model.py first")
        sys.exit(1)

    print("📡 Loading model...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']

    arduino_port = "/dev/ttyUSB0"
    try:
        ser = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)
    except serial.SerialException:
        print(f"❌ Could not open {arduino_port}")
        print("   Check your Arduino connection")
        print("   Common ports: /dev/ttyUSB0, /dev/ttyACM0, COM3")
        sys.exit(1)

    print("🔥 LIVE FOREST FIRE PREDICTION")
    print("=" * 40)
    print(f"Reading from: {arduino_port}")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            line = ser.readline().decode().strip()
            if not line or line == "ERROR":
                continue

            parts = line.split(",")
            if len(parts) != 2:
                continue

            temp = float(parts[0])
            hum = float(parts[1])

            X = scaler.transform([[temp, hum]])
            prob = model.predict_proba(X)[0, 1]

            risk_level, status = get_risk_level(prob)
            risk_bar = "█" * int(prob * 20)

            print(f"Temp: {temp:5.1f}°C | Hum: {hum:5.1f}%")
            print(f"Risk: [{risk_bar:<20}] {prob:.1%}")
            print(f"Status: {risk_level} - {status}")
            print("-" * 40)

        except (ValueError, IndexError):
            continue
        except KeyboardInterrupt:
            print("\n👋 Stopped")
            break


if __name__ == "__main__":
    main()