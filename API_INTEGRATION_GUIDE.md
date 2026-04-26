# API Integration Guide

## Unified Prediction System (`src/inference/predictor.py`)

### Key Components

#### 1. **Interfaces & Input Classes**

```
PredictionInput (Abstract Base Class)
├── ImagePredictionInput: For image-based fire detection
└── SensorPredictionInput: For sensor-based fire detection
```

Each input class validates its data before prediction:
- `ImagePredictionInput`: Validates file exists, checks image format
- `SensorPredictionInput`: Validates temperature range (-50 to 60°C), humidity (0-100%)

#### 2. **PredictionResult Data Class**

Returns structured prediction output:
```python
@dataclass
class PredictionResult:
    prediction: int           # 0: no fire, 1: fire
    confidence: float         # 0-1 probability
    risk_level: str          # LOW, MEDIUM, HIGH, CRITICAL
    metadata: Dict[str, Any] # Additional context
```

#### 3. **FirePredictionSystem Class**

Main interface for making predictions:

```python
# Initialize with both models
system = FirePredictionSystem(
    cnn_model_path='models/saved/forest_fire_cnn_final.pth',
    sensor_model_path='models/saved/sensor_model.pkl',
    device='cpu'  # or 'cuda'
)

# Image prediction
image_input = ImagePredictionInput('path/to/image.jpg')
result = system.predict_from_image(image_input)

# Sensor prediction
sensor_input = SensorPredictionInput(temperature=35.5, humidity=25.0)
result = system.predict_from_sensors(sensor_input)

# Ensemble prediction (weighted average)
result = system.combined_prediction(
    image_input=image_input,
    sensor_input=sensor_input,
    image_weight=0.6,
    sensor_weight=0.4
)
```

---

## Arduino Reader (`arduino_reader.py`)

### Interfaces & Classes

```
DataSource (Abstract Base Class)
├── connect(): Connect to data source
├── read(): Read data from source
└── disconnect(): Close connection
    └── ArduinoDataSource (Implementation)
        - Communicates via serial port
        - Parses "T:25.5,H:45.2" format
```

### FireRiskMonitor Class

Real-time monitoring with automatic risk assessment:

```python
from arduino_reader import ArduinoDataSource, FireRiskMonitor

# Initialize
arduino_source = ArduinoDataSource(
    port='/dev/ttyUSB0',
    baudrate=9600
)

monitor = FireRiskMonitor(
    data_source=arduino_source,
    sensor_model_path='models/saved/sensor_model.pkl',
    update_interval=5.0  # seconds
)

# Start monitoring (blocks until interrupted)
monitor.start()

# Get statistics after monitoring
stats = monitor.get_statistics()
```

**Features:**
- Color-coded console output (Green/Yellow/Red/Magenta)
- Automatic history saving to JSON
- Statistics aggregation
- Graceful error handling

---

## Flask Web Dashboard (`app.py`)

### Routes

#### Dashboard
- **GET** `/` - Main dashboard page

#### API Endpoints

**Sensor Readings:**
- **GET** `/api/readings?limit=20` - Get recent sensor readings
- **POST** `/api/readings` - Add new sensor reading
  ```json
  {
    "temperature": 25.5,
    "humidity": 45.2
  }
  ```

**Image Prediction:**
- **POST** `/api/predict-image` - Predict fire from uploaded image
  - Form data: `image` (multipart file)

**Ensemble Prediction:**
- **POST** `/api/predict-ensemble` - Combined image + sensor prediction
  - Form data: `image` (multipart), `temperature`, `humidity`

**Model Information:**
- **GET** `/api/model-info` - Get loaded model details

**Statistics:**
- **GET** `/api/statistics` - Get readings statistics

### Running the Dashboard

```bash
# Install requirements
pip install -r requirements.txt

# Run Flask app
python app.py
# Accessible at http://localhost:5000
```

---

## Model Evaluation (`evaluate_models.py`)

Comprehensive evaluation script generating:
- Confusion matrices
- ROC curves
- Precision-recall curves
- Performance comparison charts
- Classification reports

```bash
python evaluate_models.py
# Outputs to models/plots/
```

---

## Configuration (`.env` file)

```bash
# Copy .env.example to .env and configure:

CNN_MODEL_PATH=models/saved/forest_fire_cnn_final.pth
SENSOR_MODEL_PATH=models/saved/sensor_model.pkl
ARDUINO_PORT=/dev/ttyUSB0
ARDUINO_BAUDRATE=9600
FLASK_PORT=5000
IMAGE_WEIGHT=0.6
SENSOR_WEIGHT=0.4
```

---

## Design Patterns Used

### 1. **Dependency Injection**
- Models loaded externally and passed to system
- Enables testing and flexibility

### 2. **Strategy Pattern**
- `PredictionInput` interface allows different input strategies
- `DataSource` interface for different data source implementations

### 3. **Factory Pattern**
- `SensorFireRiskModel._create_model()` - Creates appropriate model type

### 4. **Data Class Pattern**
- `PredictionResult` - Immutable result container

### 5. **Singleton Pattern (implicit)**
- `FirePredictionSystem` - Single instance managing both models
- `FireRiskMonitor` - Single monitoring session

---

## Error Handling

All components implement comprehensive error handling:

```python
try:
    result = system.predict_from_image(image_input)
except FileNotFoundError:
    # Handle missing image file
except ValueError:
    # Handle invalid input
except RuntimeError:
    # Handle prediction failure
```

---

## Type Hints

All functions include full type hints for IDE support and type checking:

```python
def predict_from_sensors(self, sensor_input: SensorPredictionInput) -> PredictionResult:
```

---

## Usage Examples

See `examples.py` for complete working examples:
```bash
python examples.py
```

---

## Performance Notes

- **CNN Model**: ~200-500ms per image (CPU)
- **Sensor Model**: <1ms per prediction
- **Ensemble**: Combined time + minimal overhead

For GPU acceleration, set `device='cuda'` in `FirePredictionSystem.__init__()`

---

## Monitoring Statistics

Real-time monitoring tracks:
- Temperature: min, max, average
- Humidity: min, max, average
- Risk distribution: count by level
- Average fire probability
