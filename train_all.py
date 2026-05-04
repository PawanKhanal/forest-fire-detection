#!/usr/bin/env python3
"""
Train all models for the Forest Fire Detection System.
This script provides a single entry point to train both the CNN model for images
and the Random Forest model for sensor data.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd_list, description):
    print(f"\n{'=' * 60}")
    print(f"🚀 Starting: {description}")
    print(f"{'=' * 60}\n")
    try:
        subprocess.run(cmd_list, check=True)
        print(f"\n✅ Successfully completed: {description}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during {description}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train all Forest Fire models")
    parser.add_argument(
        "--cnn-only", action="store_true", help="Only train the CNN image model"
    )
    parser.add_argument(
        "--sensor-only", action="store_true", help="Only train the sensor data model"
    )
    
    # CNN args
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for CNN")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CNN")

    args = parser.parse_args()

    # Determine what to run
    run_cnn = True
    run_sensor = True
    if args.cnn_only:
        run_sensor = False
    elif args.sensor_only:
        run_cnn = False

    if run_sensor:
        run_command(
            [sys.executable, "train_sensor_model.py"],
            "Sensor Model Training (Random Forest & Nepal Calibrated Predictor)"
        )

    if run_cnn:
        run_command(
            [
                sys.executable, 
                "train_pytorch.py", 
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size)
            ],
            "Image Model Training (PyTorch CNN)"
        )

    print(f"\n{'=' * 60}")
    print("🎉 All requested training processes completed successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
