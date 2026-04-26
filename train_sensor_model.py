#!/usr/bin/env python3
"""Train sensor-based ML model on Portuguese Forest Fire Dataset + Arduino data."""

import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from models.sensor_model import SensorFireRiskModel


def prepare_sensor_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features from sensor DataFrame."""
    X = df[['temp', 'RH']].values
    y = (df['area'] > 0).astype(int).values
    return X, y


def prepare_arduino_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features from Arduino sensor data."""
    X = df[['temperature', 'humidity']].values
    y = df['fire_risk'].values
    return X, y


def plot_training_curves(model, X, y, save_dir: Path):
    """Generate learning curves."""
    print("\n📈 Generating Learning Curves...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Accuracy')
    plt.fill_between(train_sizes, val_mean - train_std, val_mean + val_std, alpha=0.1, color='green')
    
    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves - Sensor Fire Risk Model')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "sensor_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved to {save_dir}/sensor_learning_curves.png")


def plot_confusion_matrix_vis(y_true, y_pred, save_dir: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'])
    plt.title('Confusion Matrix - Sensor Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "sensor_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_data_patterns(df: pd.DataFrame):
    """Analyze what conditions lead to fire."""
    print("\n🔍 Data Pattern Analysis:")
    print("-" * 40)
    
    if 'area' in df.columns:
        fire_df = df[df['area'] > 0]
        no_fire_df = df[df['area'] == 0]
        print(f"\n📊 Fire Days (n={len(fire_df)}):")
        print(f"   Temperature: {fire_df['temp'].mean():.1f}°C (±{fire_df['temp'].std():.1f})")
        print(f"   Humidity: {fire_df['RH'].mean():.1f}% (±{fire_df['RH'].std():.1f})")
        print(f"\n📊 Non-Fire Days (n={len(no_fire_df)}):")
        print(f"   Temperature: {no_fire_df['temp'].mean():.1f}°C (±{no_fire_df['temp'].std():.1f})")
        print(f"   Humidity: {no_fire_df['RH'].mean():.1f}% (±{no_fire_df['RH'].std():.1f})")
    else:
        fire_df = df[df['fire_risk'] == 1]
        no_fire_df = df[df['fire_risk'] == 0]
        print(f"\n📊 Fire Risk=1 (n={len(fire_df)}):")
        print(f"   Temperature: {fire_df['temperature'].mean():.1f}°C")
        print(f"   Humidity: {fire_df['humidity'].mean():.1f}%")
        print(f"\n📊 Fire Risk=0 (n={len(no_fire_df)}):")
        print(f"   Temperature: {no_fire_df['temperature'].mean():.1f}°C")
        print(f"   Humidity: {no_fire_df['humidity'].mean():.1f}%")


def main():
    print("🔥 Training Sensor-Based Fire Risk Model")
    print("=" * 50)
    print("Dataset: Portuguese Forest Fires + Arduino Data")
    print("Model: Random Forest Classifier\n")
    
    data_dir = Path("data/sensor")
    save_dir = Path("models/plots")
    model_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    
    forestfires_path = data_dir / "forestfires.csv"
    if forestfires_path.exists():
        print("📥 Loading UCI Portuguese Fire Dataset...")
        df_uci = pd.read_csv(forestfires_path)
        dfs.append(df_uci)
        print(f"   Loaded {len(df_uci)} samples")
    
    arduino_path = data_dir / "arduino_training_data.csv"
    if arduino_path.exists():
        print("📥 Loading Arduino Training Data...")
        df_arduino = pd.read_csv(arduino_path)
        dfs.append(df_arduino)
        print(f"   Loaded {len(df_arduino)} samples")
    
    if not dfs:
        print("❌ No data found!")
        print("   Place CSV files in data/sensor/")
        sys.exit(1)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total samples: {len(df)}")
    
    if 'temp' in df.columns and 'RH' in df.columns:
        X, y = prepare_sensor_data(df)
    else:
        X, y = prepare_arduino_data(df)
    
    fire_count = y.sum()
    print(f"   Fire cases: {fire_count} ({y.mean()*100:.1f}%)")
    print(f"   Non-fire: {len(y) - fire_count} ({(1-y.mean())*100:.1f}%)")
    
    analyze_data_patterns(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n🔥 Training Random Forest...")
    start_time = time.time()
    
    model = SensorFireRiskModel(model_type='random_forest')
    model.model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    y_pred = model.model.predict(X_test_scaled)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    print(f"\n✅ Completed in {training_time:.4f} seconds")
    print("\n📈 Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f} ({value*100:.1f}%)")
    
    print("\n🔄 5-Fold Cross-Validation:")
    X_scaled_full = scaler.fit_transform(X)
    cv_scores = cross_val_score(model.model, X_scaled_full, y, cv=5, scoring='accuracy')
    print(f"   Scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"   Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    plot_confusion_matrix_vis(y_test, y_pred, save_dir)
    
    print("\n🔥 Training final model on full dataset...")
    model.model.fit(X_scaled_full, y)
    
    plot_training_curves(model.model, X_scaled_full, y, save_dir)
    
    save_path = model_dir / "sensor_model.pkl"
    joblib.dump({
        'model': model.model,
        'scaler': scaler,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }, save_path)
    print(f"\n💾 Model saved to {save_path}")
    
    print("\n🧪 Test Scenarios:")
    print("-" * 50)
    test_scenarios = [
        (12.0, 85.0, "Cool, very humid"),
        (20.0, 55.0, "Mild conditions"),
        (28.0, 30.0, "Warm, dry"),
        (33.0, 18.0, "Hot, very dry"),
    ]
    
    for temp, hum, desc in test_scenarios:
        input_scaled = scaler.transform([[temp, hum]])
        proba = model.model.predict_proba(input_scaled)[0, 1]
        pred = model.model.predict(input_scaled)[0]
        
        if proba < 0.35:
            risk_level = "LOW"
        elif proba < 0.50:
            risk_level = "MODERATE"
        elif proba < 0.65:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        risk_bar = "█" * int(proba * 20)
        
        print(f"\n{desc}: {temp}°C, {hum}% humidity")
        print(f"  Risk: {'⚠️ FIRE' if pred == 1 else '✅ SAFE'} | Level: {risk_level}")
        print(f"  Probability: [{risk_bar:<20}] {proba:.1%}")
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print(f"   Accuracy: {metrics['Accuracy']*100:.1f}%")
    print(f"   Ready for Arduino: python arduino/live_predict.py")


if __name__ == "__main__":
    main()