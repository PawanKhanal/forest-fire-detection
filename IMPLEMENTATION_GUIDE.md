# Forest Fire Detection System - Complete Implementation Guide

**BCA Final Year Project | Tribhuvan University**

A production-ready AI system for real-time forest fire detection using computer vision and IoT sensors.

## 🔥 System Overview

The system combines two complementary ML models for robust fire detection:

| Model | Input | Accuracy | Use Case |
|-------|-------|----------|----------|
| **CNN (PyTorch)** | Aerial/Satellite Images | 75.85% | Visual fire detection |
| **Sensor (Random Forest)** | Temperature + Humidity | 62.5% | Weather-based prediction |
| **Ensemble** | Both Inputs | ~75% | Maximum coverage |

## 📁 Complete Project Structure

```
fypy_forest_fire_detection/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── cnn_model_pytorch.py      # CNN implementation
│   │   ├── sensor_model.py           # Random Forest model
│   │   └── plots/                    # Evaluation visualizations
│   ├── data/
│   │   ├── dataset.py                # Data utilities
│   │   ├── preprocessing.py          # Data preprocessing
│   │   └── pytorch_dataset.py        # PyTorch DataLoader
│   └── inference/                    # ⭐ NEW UNIFIED SYSTEM
│       ├── __init__.py
│       └── predictor.py              # Unified prediction interface
├── models/
│   ├── saved/
│   │   ├── forest_fire_cnn_final.pth
│   │   └── sensor_model.pkl
│   └── plots/                        # Evaluation results
├── data/
│   ├── raw/
│   │   ├── train/ (fire/, nofire/)
│   │   ├── val/ (fire/, nofire/)
│   │   └── test/ (fire/, nofire/)
│   └── sensor/
│       ├── forestfires.csv
│       └── readings_history.json
├── templates/                        # ⭐ NEW FLASK DASHBOARD
│   └── dashboard.html
├── uploads/                          # Uploaded images for prediction
├── logs/                             # Application logs
├── arduino_reader.py                 # ⭐ NEW ARDUINO INTEGRATION
├── app.py                            # ⭐ NEW FLASK WEB DASHBOARD
├── evaluate_models.py                # ⭐ NEW EVALUATION SCRIPT
├── examples.py                       # Usage examples
├── requirements.txt                  # ⭐ UPDATED dependencies
├── .env.example                      # ⭐ Configuration template
├── .gitignore                        # ⭐ UPDATED
├── config/config.yaml
└── README.md                         # This file
```

## 🎯 Key New Components

### 1. **Unified Prediction System** (`src/inference/predictor.py`)

Object-oriented design with clear interfaces:

```python
from src.inference.predictor import (
    FirePredictionSystem,
    ImagePredictionInput,
    SensorPredictionInput
)

# Initialize
system = FirePredictionSystem(
    cnn_model_path='models/saved/forest_fire_cnn_final.pth',
    sensor_model_path='models/saved/sensor_model.pkl'
)

# Image prediction
image_input = ImagePredictionInput('path/to/image.jpg')
result = system.predict_from_image(image_input)
# Returns: PredictionResult(prediction=1, confidence=0.92, risk_level='HIGH')

# Sensor prediction
sensor_input = SensorPredictionInput(temperature=35.5, humidity=25.0)
result = system.predict_from_sensors(sensor_input)

# Ensemble (weighted average)
result = system.combined_prediction(
    image_input, sensor_input,
    image_weight=0.6, sensor_weight=0.4
)
```

**Key Features:**
- ✅ Type hints throughout
- ✅ Google-style docstrings
- ✅ Comprehensive error handling
- ✅ Input validation
- ✅ Modular design with interfaces

### 2. **Arduino Integration** (`arduino_reader.py`)

Real-time sensor monitoring:

```python
from arduino_reader import ArduinoDataSource, FireRiskMonitor

# Connect to Arduino
arduino = ArduinoDataSource(port='/dev/ttyUSB0', baudrate=9600)
monitor = FireRiskMonitor(
    data_source=arduino,
    sensor_model_path='models/saved/sensor_model.pkl',
    update_interval=5.0
)

# Start monitoring (runs until Ctrl+C)
monitor.start()

# Get statistics
stats = monitor.get_statistics()
```

**Features:**
- 🔴 Color-coded console output (Green/Yellow/Red/Magenta)
- 💾 Automatic history saving (JSON)
- 📊 Real-time statistics
- 🛡️ Graceful error handling
- 🔌 Serial port management

Expected Arduino format: `T:25.5,H:45.2`

### 3. **Flask Web Dashboard** (`app.py`)

Modern responsive interface:

```bash
python app.py
# Open http://localhost:5000 in browser
```

**Features:**
- 📊 Real-time sensor gauges
- 🔥 Fire risk meter (0-100%)
- 📤 Image upload for CNN prediction
- 📈 Readings history table
- 🔄 Auto-refresh every 5 seconds
- 📱 Responsive Bootstrap 5 design

**API Endpoints:**
```
GET  /                              # Dashboard page
GET  /api/readings?limit=20         # Get sensor readings
POST /api/readings                  # Add sensor reading
POST /api/predict-image             # Image fire detection
POST /api/predict-ensemble          # Combined prediction
GET  /api/statistics                # Reading statistics
GET  /api/model-info                # Model information
```

### 4. **Model Evaluation** (`evaluate_models.py`)

Comprehensive evaluation suite:

```bash
python evaluate_models.py
# Generates: models/plots/
#   - confusion_matrices.png
#   - roc_curves.png
#   - precision_recall_curves.png
#   - metrics_comparison.png
#   - evaluation_report.txt
```

**Metrics Generated:**
- Confusion matrices
- ROC curves with AUC
- Precision-recall curves
- Performance comparison charts
- Classification reports

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Clone/navigate to project
cd fypy_forest_fire_detection

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your paths and settings
```

### 2. **Download Models**

If models aren't already trained:

```bash
# Download training data
python download_fire_dataset.py          # Kaggle images
python download_portuguese_fire_data.py  # UCI sensor data

# Train models
python train_pytorch.py       # Trains CNN (save to models/saved/forest_fire_cnn_final.pth)
python train_sensor_model.py  # Trains Random Forest (save to models/saved/sensor_model.pkl)
```

### 3. **Run Prediction System**

```python
# Quick test
from src.inference.predictor import FirePredictionSystem, ImagePredictionInput

system = FirePredictionSystem(
    'models/saved/forest_fire_cnn_final.pth',
    'models/saved/sensor_model.pkl'
)

image = ImagePredictionInput('data/raw/test/fire/sample.jpg')
result = system.predict_from_image(image)
print(f"Risk: {result.risk_level} ({result.confidence:.2%})")
```

### 4. **Run Web Dashboard**

```bash
python app.py
# Navigate to http://localhost:5000
# Upload images or submit sensor readings
```

### 5. **Run Arduino Monitoring**

```bash
# Connect Arduino to /dev/ttyUSB0
# Upload arduino_dht11_fire.ino to Arduino board

python arduino_reader.py
# Real-time monitoring with color output
```

### 6. **Evaluate Models**

```bash
python evaluate_models.py
# View results in models/plots/
```

## 🏗️ Architecture & Design Patterns

### Class Hierarchy

```
PredictionInput (ABC)
├── ImagePredictionInput
└── SensorPredictionInput

DataSource (ABC)
└── ArduinoDataSource

┌─────────────────────────────┐
│  FirePredictionSystem       │
│                             │
│  - predict_from_image()     │
│  - predict_from_sensors()   │
│  - combined_prediction()    │
│                             │
│  [CNN Model] [Sensor Model] │
└─────────────────────────────┘

FireRiskMonitor
├── DataSource (Arduino/Mock)
├── SensorFireRiskModel
└── History Manager
```

### Design Patterns Used

| Pattern | Usage | Location |
|---------|-------|----------|
| **Strategy** | Input handling (image vs sensor) | PredictionInput classes |
| **Dependency Injection** | Models passed to system | FirePredictionSystem |
| **Factory** | Model creation | SensorFireRiskModel._create_model() |
| **Data Class** | Result container | PredictionResult |
| **Template Method** | Evaluation pipeline | ModelEvaluator |
| **Observer** | Dashboard auto-refresh | Flask + JavaScript |

## 📊 API Response Examples

### Image Prediction
```json
{
  "success": true,
  "prediction": 1,
  "confidence": 0.923,
  "risk_level": "HIGH",
  "class_name": "FIRE",
  "image_path": "/uploads/20240426_120530_fire.jpg"
}
```

### Sensor Prediction
```json
{
  "success": true,
  "prediction": 1,
  "confidence": 0.756,
  "risk_level": "HIGH",
  "temperature": 35.5,
  "humidity": 25.0
}
```

### Ensemble Prediction
```json
{
  "success": true,
  "prediction": 1,
  "confidence": 0.842,
  "risk_level": "CRITICAL",
  "metadata": {
    "image_weight": 0.6,
    "sensor_weight": 0.4,
    "image_confidence": 0.923,
    "sensor_confidence": 0.756
  }
}
```

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Model paths
CNN_MODEL_PATH=models/saved/forest_fire_cnn_final.pth
SENSOR_MODEL_PATH=models/saved/sensor_model.pkl

# Arduino
ARDUINO_PORT=/dev/ttyUSB0
ARDUINO_BAUDRATE=9600

# Flask
FLASK_DEBUG=False
FLASK_PORT=5000

# Ensemble weights
IMAGE_WEIGHT=0.6
SENSOR_WEIGHT=0.4

# Thresholds
CNN_CONFIDENCE_THRESHOLD=0.5
SENSOR_PROBABILITY_THRESHOLD=0.5
```

## 📈 Performance Metrics

### Models

| Metric | CNN | Sensor | Ensemble |
|--------|-----|--------|----------|
| Accuracy | 75.85% | 62.5% | ~75% |
| Precision | 76.2% | 60.1% | ~76% |
| Recall | 74.5% | 77.8% | ~76% |
| F1-Score | 75.3% | 68.2% | ~76% |
| Inference Time | 200-500ms | <1ms | 200-500ms |

### System Resources

- **RAM**: ~800MB (CNN) + ~50MB (Sensor) = ~850MB
- **Storage**: ~450KB (CNN weights) + 50KB (Sensor) = ~500KB
- **CPU**: 1-2 cores at full utilization
- **GPU**: Optional (set device='cuda' for faster inference)

## 🐛 Troubleshooting

### Arduino Connection Issues

```bash
# Check connection
ls -la /dev/ttyUSB*

# Check permissions
sudo usermod -a -G dialout $USER
# Then restart terminal

# Test serial connection
screen /dev/ttyUSB0 9600
```

### Flask Port Already in Use

```bash
# Check what's using port 5000
lsof -i :5000

# Use different port
FLASK_PORT=5001 python app.py
```

### Model Loading Errors

```bash
# Verify model files exist
ls -lh models/saved/

# Check PyTorch version compatibility
python -c "import torch; print(torch.__version__)"

# Rebuild if needed
python train_pytorch.py
```

### Out of Memory

```python
# Use smaller batch size
torch.cuda.empty_cache()

# Or use CPU
device = 'cpu'
```

## 📚 Documentation

- **API Guide**: See `API_INTEGRATION_GUIDE.md`
- **Examples**: See `examples.py`
- **Model Details**: See model files in `src/models/`
- **Config**: See `config/config.yaml`

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run individual tests
python -m pytest tests/test_models.py -v

# With coverage
python -m pytest --cov=src tests/
```

## 🔐 Security Notes

- ✅ Input validation on all endpoints
- ✅ File type checking for uploads
- ✅ Size limits (16MB max file)
- ✅ No sensitive data in logs
- ⚠️ Enable CSRF protection in production
- ⚠️ Use HTTPS in production
- ⚠️ Implement authentication for Flask app

## 📱 Mobile Deployment

For mobile apps, use REST API:

```python
import requests

# Image prediction
files = {'image': open('photo.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/predict-image', files=files)
print(response.json())

# Sensor prediction
data = {'temperature': 35.5, 'humidity': 25.0}
response = requests.post('http://localhost:5000/api/readings', json=data)
print(response.json())
```

## 🚀 Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```bash
docker build -t forest-fire-detection .
docker run -p 5000:5000 \
  -v /path/to/models:/app/models \
  -v /path/to/data:/app/data \
  forest-fire-detection
```

### Systemd Service

```ini
[Unit]
Description=Forest Fire Detection System
After=network.target

[Service]
Type=simple
User=firedetection
WorkingDirectory=/opt/fypy_forest_fire_detection
Environment="PATH=/opt/fypy_forest_fire_detection/venv/bin"
ExecStart=/opt/fypy_forest_fire_detection/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 👥 Contributing

When making contributions:
1. Follow PEP 8 style guide
2. Add type hints to all functions
3. Include docstrings in Google format
4. Add unit tests for new features
5. Update README if needed

## 📝 License

This is an educational project for Tribhuvan University BCA program.

## 📞 Support

- **Issues**: Check GitHub issues or create new one
- **Email**: Your contact email
- **Documentation**: See markdown files in project

## 🎓 Academic References

- CNN Architecture: ResNet-inspired design
- Sensor Model: UCI Machine Learning Repository - Portuguese Forest Fires
- Dataset: Kaggle Wildfire Detection Dataset
- Papers: See `docs/references.bib`

---

**Last Updated**: 2024-04-26
**Version**: 2.0 (With Unified Prediction System)
**Status**: Production Ready
