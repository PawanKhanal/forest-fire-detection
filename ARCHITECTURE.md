"""Architecture and OOP Design Patterns Documentation"""

# FOREST FIRE DETECTION SYSTEM - OOP ARCHITECTURE

## 1. INTERFACE-BASED DESIGN

### Input Abstraction Layer
```
┌─────────────────────────────────────────┐
│   PredictionInput (ABC)                 │
│   ├── validate(): bool                  │
│   ├── get_input_type(): str             │
│   └── Abstract methods                  │
└─────────────────────────────────────────┘
    ▲               ▲
    │               │
┌───┴────────┐  ┌──┴──────────────┐
│   ImagePredictionInput          │
│   - image_path: Path            │
│   - image_size: tuple           │
│   - validate(): Check file      │
│   - load_image(): Tensor        │
│   - get_input_type(): "image"   │
└─────────────────────────────────┘

│   SensorPredictionInput         │
│   - temperature: float          │
│   - humidity: float             │
│   - validate(): Check ranges    │
│   - get_input_type(): "sensor"  │
└─────────────────────────────────┘
```

**Key Benefits:**
- Clear contract for different input types
- Type-safe predictions
- Extensible for future input types (video, stream, etc.)
- Compile-time errors vs runtime surprises

---

## 2. DATA SOURCE ABSTRACTION LAYER

### Polymorphic Data Sources
```
┌─────────────────────────────────────────┐
│   DataSource (ABC)                      │
│   ├── connect(): bool                   │
│   ├── read(): Optional[Dict]            │
│   ├── disconnect(): void                │
│   └── Abstract interface                │
└─────────────────────────────────────────┘
    ▲
    │
    └─ ArduinoDataSource (Concrete Implementation)
       ├── port: str
       ├── baudrate: int
       ├── serial_conn: Serial
       ├── connect(): Opens serial port
       ├── read(): Parses "T:X,H:Y"
       ├── disconnect(): Closes connection
       └── _parse_arduino_data(): Validates format
```

**Extensibility Example:**
```python
# Future: Add more data sources without changing FireRiskMonitor
class MockDataSource(DataSource):
    def connect(self): return True
    def read(self): return {'temperature': 30.0, 'humidity': 50.0}
    def disconnect(self): pass

class CloudDataSource(DataSource):
    def connect(self): # Connect to API
    def read(self): # Fetch from cloud
    def disconnect(self): # Clean up

# Use any implementation
monitor = FireRiskMonitor(
    data_source=ArduinoDataSource(),  # or MockDataSource() or CloudDataSource()
    sensor_model_path='...'
)
```

---

## 3. PREDICTION SYSTEM - UNIFIED INTERFACE

### Dependency Injection Pattern
```
FirePredictionSystem
├── Dependencies Injected:
│   ├── CNN Model: ForestFireCNN
│   ├── Sensor Model: SensorFireRiskModel
│   ├── Device: str ('cpu' or 'cuda')
│   └── Image Size: tuple
│
├── Input Types Accepted:
│   ├── ImagePredictionInput
│   ├── SensorPredictionInput
│   └── Both (for ensemble)
│
├── Methods:
│   ├── predict_from_image(ImagePredictionInput) → PredictionResult
│   ├── predict_from_sensors(SensorPredictionInput) → PredictionResult
│   └── combined_prediction(...) → PredictionResult
│
└── Output:
    └── PredictionResult
        ├── prediction: int
        ├── confidence: float
        ├── risk_level: str
        └── metadata: Dict
```

**Key Principle:** System depends on abstractions, not concrete implementations.

---

## 4. DATA FLOW ARCHITECTURE

### Image-Based Pipeline
```
User selects image
    ↓
ImagePredictionInput(path)
    ├── validate() → Check file exists & format
    ├── load_image() → PIL → Tensor
    └── (inherits from PredictionInput)
    ↓
FirePredictionSystem.predict_from_image()
    ├── Load image tensor
    ├── Run through CNN
    ├── Apply softmax
    ├── Classify & confidence
    └── Map to risk level
    ↓
PredictionResult
    ├── prediction: int (0=no fire, 1=fire)
    ├── confidence: float (0-1)
    ├── risk_level: str (LOW/MED/HIGH/CRITICAL)
    └── metadata: {class_name, device, ...}
    ↓
Dashboard Display
    └── Show prediction with confidence
```

### Sensor-Based Pipeline
```
Arduino sends: "T:35.5,H:25.0"
    ↓
ArduinoDataSource.read()
    ├── Read serial line
    ├── _parse_arduino_data()
    └── Return {temperature, humidity}
    ↓
SensorPredictionInput(temp, humidity)
    ├── validate() → Check ranges (-50 to 60°C, 0-100%)
    └── (inherits from PredictionInput)
    ↓
FirePredictionSystem.predict_from_sensors()
    ├── Create feature array [temp, humidity]
    ├── Run through Random Forest
    ├── Get probability
    └── Map to risk level
    ↓
PredictionResult
    ├── prediction: int (0=no fire, 1=fire)
    ├── confidence: float (probability)
    ├── risk_level: str (LOW/MED/HIGH/CRITICAL)
    └── metadata: {temperature, humidity, ...}
    ↓
Real-Time Monitor Display
    ├── Color-coded output
    ├── Save to history
    └── Update statistics
```

### Ensemble Pipeline
```
Image + Sensor Inputs
    ├── ImagePredictionInput
    └── SensorPredictionInput
    ↓
FirePredictionSystem.combined_prediction()
    ├── Run predict_from_image() → conf_img
    ├── Run predict_from_sensors() → conf_sensor
    ├── Weighted average:
    │   ensemble_conf = (conf_img × 0.6) + (conf_sensor × 0.4)
    ├── Classify: if ensemble_conf ≥ 0.5 → fire, else no fire
    └── Map to risk level
    ↓
PredictionResult (Ensemble)
    └── metadata: {
        ensemble_type, weights,
        image_confidence, sensor_confidence,
        full_results_for_both
    }
    ↓
Dashboard Display
    └── Show combined risk with component breakdown
```

---

## 5. CLASS RESPONSIBILITIES (Single Responsibility Principle)

| Class | Responsibility |
|-------|-----------------|
| **PredictionInput** | Define contract for input validation |
| **ImagePredictionInput** | Validate & load image files |
| **SensorPredictionInput** | Validate sensor value ranges |
| **DataSource** | Define contract for data sources |
| **ArduinoDataSource** | Communicate with Arduino via serial |
| **FirePredictionSystem** | Orchestrate both models & make predictions |
| **SensorFireRiskModel** | Random Forest model predictions |
| **ForestFireCNN** | CNN model architecture & inference |
| **PredictionResult** | Package results in structured format |
| **FireRiskMonitor** | Real-time monitoring & history |
| **ModelEvaluator** | Performance evaluation & visualization |

---

## 6. DESIGN PATTERNS IMPLEMENTED

### A. Strategy Pattern
```python
# Different input strategies - same prediction system
def predict(input: PredictionInput):
    # Strategy varies based on input type
    if isinstance(input, ImagePredictionInput):
        # Strategy 1: CNN pipeline
    elif isinstance(input, SensorPredictionInput):
        # Strategy 2: RandomForest pipeline

# Better: System handles polymorphism
result = system.predict_from_image(ImagePredictionInput(...))
result = system.predict_from_sensors(SensorPredictionInput(...))
```

### B. Dependency Injection
```python
# BAD: Tight coupling
class FirePredictionSystem:
    def __init__(self):
        self.cnn = ForestFireCNN.load('path')  # Hardcoded
        self.sensor = SensorFireRiskModel.load('path')  # Hardcoded

# GOOD: Dependencies injected
class FirePredictionSystem:
    def __init__(self, cnn_path: str, sensor_path: str):
        self.cnn = ForestFireCNN.load(cnn_path)  # Configurable
        self.sensor = SensorFireRiskModel.load(sensor_path)  # Configurable
```

### C. Factory Pattern
```python
class SensorFireRiskModel:
    def _create_model(self):
        models = {
            'random_forest': RandomForestClassifier(...),
            'logistic': LogisticRegression(...)
        }
        return models.get(self.model_type)
```

### D. Template Method Pattern
```python
class ModelEvaluator:
    def evaluate_all(self):
        self.load_models()          # Step 1
        cnn_metrics = self.evaluate_cnn_model()  # Step 2
        sensor_metrics = self.evaluate_sensor_model()  # Step 2
        self.plot_confusion_matrices(...)  # Step 3
        self.plot_roc_curves(...)   # Step 3
        # etc...
```

### E. Decorator Pattern (Flask)
```python
@require_prediction_system  # Decorator checks system initialized
def predict_image() -> Dict[str, Any]:
    if prediction_system is None:
        return error response
    return result
```

---

## 7. EXTENSIBILITY EXAMPLES

### Adding New Input Type (Video)
```python
class VideoPredictionInput(PredictionInput):
    """New input type for video files"""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
    
    def validate(self) -> bool:
        # Validate video format
        return True
    
    def get_input_type(self) -> str:
        return "video"
    
    def extract_frames(self) -> List[np.ndarray]:
        # Extract frames from video
        pass

# Use with same system
system = FirePredictionSystem(...)
video_input = VideoPredictionInput('video.mp4')
result = system.predict_from_video(video_input)  # Could add this method
```

### Adding New Data Source (Cloud API)
```python
class CloudSensorDataSource(DataSource):
    """Get sensor data from cloud IoT platform"""
    
    def connect(self) -> bool:
        # Authenticate with cloud API
        pass
    
    def read(self) -> Optional[Dict]:
        # Fetch latest sensor reading from cloud
        pass
    
    def disconnect(self) -> None:
        # Close connection
        pass

# Use with same monitor
cloud_source = CloudSensorDataSource()
monitor = FireRiskMonitor(
    data_source=cloud_source,
    sensor_model_path='...'
)
monitor.start()
```

---

## 8. ERROR HANDLING HIERARCHY

```
Exception
├── FileNotFoundError
│   └── model_path not found
│   └── image_path not found
│
├── ValueError
│   ├── temperature out of range (-50 to 60°C)
│   ├── humidity out of range (0-100%)
│   ├── invalid image format
│   └── weights don't sum to 1.0
│
├── TypeError
│   ├── temperature not numeric
│   └── humidity not numeric
│
├── serial.SerialException
│   ├── Arduino port not found
│   └── Failed to open serial connection
│
└── RuntimeError
    ├── Model loading failed
    ├── Image prediction failed
    └── Sensor prediction failed
```

---

## 9. TYPE SAFETY

### Before (No Type Hints)
```python
def predict(self, data):  # What is data?
    result = self.model.predict(data)  # What format?
    return result  # What does this contain?
```

### After (Full Type Hints)
```python
def predict_from_image(self, image_input: ImagePredictionInput) -> PredictionResult:
    """
    Predict fire presence from image.
    
    Args:
        image_input: Validated image input
        
    Returns:
        Structured prediction result
        
    Raises:
        ValueError: If input invalid
        RuntimeError: If prediction fails
    """
```

**Benefits:**
- IDE autocomplete
- Mypy type checking
- Self-documenting code
- Early error detection

---

## 10. TESTING STRATEGY

```python
# Unit test: Image input validation
def test_image_input_validation():
    input = ImagePredictionInput('nonexistent.jpg')
    with pytest.raises(FileNotFoundError):
        input.validate()

# Unit test: Sensor input validation
def test_sensor_input_validation():
    input = SensorPredictionInput(temperature=100, humidity=50)
    with pytest.raises(ValueError):  # Out of range
        input.validate()

# Integration test: Full prediction
def test_image_prediction():
    system = FirePredictionSystem(...)
    result = system.predict_from_image(ImagePredictionInput(...))
    assert isinstance(result, PredictionResult)
    assert 0 <= result.confidence <= 1
    assert result.risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

# Integration test: Ensemble
def test_ensemble_prediction():
    system = FirePredictionSystem(...)
    result = system.combined_prediction(...)
    assert result.metadata['image_weight'] == 0.6
    assert result.metadata['sensor_weight'] == 0.4
```

---

## 11. DEPLOYMENT ARCHITECTURE

```
┌─────────────────────────────────────┐
│   Flask Web App (app.py)            │
│   - Handles HTTP requests           │
│   - Serves dashboard UI             │
│   - Manages file uploads            │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌───────────────┐ ┌──────────────────┐
│  Image Route  │ │  Sensor Route    │
│  POST image   │ │  POST readings   │
└───────┬───────┘ └────────┬─────────┘
        │                  │
        └──────────┬───────┘
                   ▼
        ┌─────────────────────┐
        │ FirePredictionSystem │
        │  - CNN              │
        │  - Sensor Model     │
        │  - Ensemble         │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
    [Models]             [Results]
    - .pth              - JSON
    - .pkl              - Database
                        - Logs
```

---

## 12. KEY PRINCIPLES FOLLOWED

✅ **DRY (Don't Repeat Yourself)**
- Common logic in base classes (PredictionInput, DataSource)

✅ **SOLID Principles**
- Single Responsibility: Each class has one reason to change
- Open/Closed: Open for extension (new input types), closed for modification
- Liskov Substitution: Any PredictionInput or DataSource works with system
- Interface Segregation: Minimal required methods in interfaces
- Dependency Inversion: Depends on abstractions, not concrete classes

✅ **KISS (Keep It Simple, Stupid)**
- Clear method names
- Minimal parameters
- Well-documented contracts

✅ **Code Reusability**
- Interfaces reduce duplication
- Models used in multiple contexts
- Utility methods extracted

---

## Summary

This architecture provides:
- 🔒 **Type Safety**: Full type hints prevent runtime errors
- 🔌 **Extensibility**: Easy to add new input types or data sources
- 🧪 **Testability**: Clear interfaces enable unit testing
- 📚 **Maintainability**: Single responsibility, clear contracts
- 🚀 **Scalability**: Ready for cloud deployment
- 🎯 **Flexibility**: Swap implementations without changing core logic
