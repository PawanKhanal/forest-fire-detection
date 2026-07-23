# Forest Fire Detection System - Technical System Implementation Details

This document contains deep technical details, structural mappings, algorithms, data flows, and configuration specifications for the Final Year Project implementation.

---

## 1. Core Model Architectures & Algorithms

### A. Deep Learning Classifier (CNN)
The system leverages transfer learning on **ResNet-18**, loaded with pre-trained weights from ImageNet.
*   **Feature Extraction**: The convolutional base remains frozen during training to conserve features like edge, gradient, and texture detectors.
*   **Custom Classification Head**: The fully connected layer is replaced with a feed-forward head that maps features to binary class logits (Fire vs. No-Fire):
    $$\text{Head}(X) = \text{Linear}(128 \to 2)\left(\text{Dropout}_{0.5}\left(\text{ReLU}\left(\text{BatchNorm1d}\left(\text{Linear}(512 \to 128)(X)\right)\right)\right)\right)$$
*   **Optimization**: Optimized using cross-entropy loss and Adam optimizer:
    $$\mathcal{L} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})$$
*   **Performance Profile**: Generates predictions within ~200-500ms on CPU.

### B. Machine Learning Environmental Model
Atmospheric data is modeled using a linear **Logistic Regression** estimator.
*   **Why Logistic Regression**: Switch from Random Forest to Logistic Regression prevents prediction plateaus at high weather margins. Logistic regression scales monotonically, classifying higher heat/dryness values with rising threat probabilities.
*   **Feature Engineering**:
    1.  **Vapor Pressure Deficit (VPD)**: Computes Saturation Vapor Pressure ($e_s$, kPa) and Actual Vapor Pressure to calculate atmosphere dryness:
        $$e_s = 0.6108 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$$
        $$\text{VPD} = e_s \times \left(1 - \frac{\text{RH}}{100}\right)$$
    2.  **Temperature-Humidity Interaction Index**:
        $$\text{TH}_{\text{interaction}} = \frac{T \times (100 - \text{RH})}{100}$$

### C. Nepal-Calibrated Environmental Predictor
Generic datasets (e.g. Portuguese agricultural burns) are mismatched with Nepal's wildfire patterns. The hybrid `NepalFirePredictor` combines physical fire weather calculations with machine learning output:
*   **Science Risk Score**:
    *   $f(T)$: Risk based on temperature brackets (higher risk as temperature exceeds 30°C).
    *   $f(\text{RH})$: Risk based on relative humidity dryness brackets (humidity below 30% indicates critical risk).
    *   $$\text{Score}_{\text{science}} = 0.5 \times f(T) + 0.5 \times f(\text{RH})$$
*   **Hybrid Blend Formula**: Science ruleset weighted at 80% and machine learning outputs weighted at 20%:
    $$\text{Risk}_{\text{final}} = \left(\text{Score}_{\text{science}} \times 0.8\right) + \left(P_{\text{ML}}(\text{Fire}) \times 0.2\right)$$
*   **Threat Categorization**:
    *   $\text{Risk}_{\text{final}} < 0.20 \implies \text{LOW}$ (Blue LED)
    *   $0.20 \le \text{Risk}_{\text{final}} < 0.50 \implies \text{MEDIUM}$ (Blue LED)
    *   $0.50 \le \text{Risk}_{\text{final}} < 0.80 \implies \text{HIGH}$ (Red LED)
    *   $\text{Risk}_{\text{final}} \ge 0.80 \implies \text{CRITICAL}$ (Red LED)

### D. Multi-Modal Ensemble
When visual satellite/UAV feed and physical hardware telemetry are both active, predictions are aggregated into a weighted average:
$$\text{Confidence}_{\text{Ensemble}} = 0.6 \times \text{Confidence}_{\text{Image}} + 0.4 \times \text{Probability}_{\text{Sensor}}$$

---

## 2. Pluggable Database Schema

The database subsystem supports both SQLite and PostgreSQL. The tables are configured as follows:

```sql
-- 1. Users Table (Role-Based Access)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    role VARCHAR(20) DEFAULT 'operator',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Sensor Readings Table
CREATE TABLE sensor_readings (
    id SERIAL PRIMARY KEY,
    timestamp VARCHAR(30) NOT NULL,
    temperature DOUBLE PRECISION NOT NULL,
    humidity DOUBLE PRECISION NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    probability DOUBLE PRECISION NOT NULL
);

-- 3. Image Predictions Table
CREATE TABLE image_predictions (
    id SERIAL PRIMARY KEY,
    timestamp VARCHAR(30) NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    risk_level VARCHAR(20) NOT NULL
);

-- 4. Ensemble Predictions Table
CREATE TABLE ensemble_predictions (
    id SERIAL PRIMARY KEY,
    timestamp VARCHAR(30) NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    temperature DOUBLE PRECISION NOT NULL,
    humidity DOUBLE PRECISION NOT NULL,
    image_confidence DOUBLE PRECISION NOT NULL,
    sensor_confidence DOUBLE PRECISION NOT NULL,
    ensemble_confidence DOUBLE PRECISION NOT NULL,
    risk_level VARCHAR(20) NOT NULL
);

-- 5. Alerts Table
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    timestamp VARCHAR(30) NOT NULL,
    source_type VARCHAR(20) NOT NULL, -- 'sensor', 'image', 'ensemble'
    source_id INTEGER NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'ACTIVE', -- 'ACTIVE', 'RESOLVED'
    resolved_by VARCHAR(50),
    resolved_at VARCHAR(30)
);
```

---

## 3. Backend REST API Reference (`app.py`)

All core dashboard endpoints (except login, status, and telemetry streams) are protected with session guards requiring active credentials.

| Endpoint | Method | Role Allowed | Description |
|---|---|---|---|
| `/` | `GET` | Public | Serves client dashboard UI. |
| `/api/auth/login` | `POST` | Public | Submits credentials and spawns session. |
| `/api/auth/logout` | `POST` | Public | Clears active cookie session. |
| `/api/auth/status` | `GET` | Public | Returns username and role for frontend rendering. |
| `/api/auth/register` | `POST` | `admin` | Registers new operator profiles. |
| `/api/readings` | `GET` | `admin`, `operator` | Fetches historical log array. |
| `/api/readings` | `POST` | Public | Endpoint for ESP32 serial/Wi-Fi telemetry logs. |
| `/api/readings/manual` | `POST` | `admin`, `operator` | Endpoint for manual dashboard submissions. |
| `/api/predict-image` | `POST` | `admin`, `operator` | Analyzes visual images. |
| `/api/predict-ensemble` | `POST` | `admin`, `operator` | Combines image and sensor telemetry. |
| `/api/alerts` | `GET` | `admin`, `operator` | Returns active environmental alarms. |
| `/api/alerts/resolve` | `POST` | `admin`, `operator` | Closes active alarm records by username. |
| `/api/statistics` | `GET` | `admin`, `operator` | Aggregates daily averages and risk frequencies. |
| `/api/model-info` | `GET` | `admin`, `operator` | Returns loaded weights metadata. |

---

## 4. Hardware Integration Details (`arduino_reader.py`)

Real-time telemetry streams from hardware microcontrollers use two connection types:

### A. Wired Serial Connection (pyserial)
*   **Data Layout**: The microcontroller outputs environmental readings via USB serial in the format `T:<temp>,H:<humidity>\n` (e.g. `T:32.4,H:28.1`).
*   **Polling Loop**: The client daemon (`arduino_reader.py`) runs an active serial thread at 9600 baud, reads raw strings, parses values, and calculates live threat probabilities.
*   **LED warning commands**: After calculation, the daemon writes the threat risk float (e.g. `0.85\n`) back to the serial connection. The microcontroller reads the risk value and triggers:
    *   **Blue LED** (GPIO 12 / Pin 8) if risk is `< 0.50` (Low/Medium risk).
    *   **Red LED** (GPIO 15 / Pin 10) if risk is `≥ 0.50` (High/Critical risk).

### B. Wireless REST Connection (Wi-Fi)
*   The ESP board utilizes the `HTTPClient` module, connects to the local LAN, and runs non-blocking HTTP POST updates to `/api/readings` every 5-10 seconds.
*   The server processes the payload, performs inference, logs the data to the SQL database, and returns the risk results in the response payload.

---

## 5. Design Patterns Applied

| Design Pattern | Purpose | Location |
|---|---|---|
| **Strategy Pattern** | Standardizes predictions for diverse data types using subclass strategies. | `PredictionInput` base class and subclasses. |
| **Dependency Injection** | Decouples systems from static files by injecting variables on initialization. | `FirePredictionSystem.__init__(cnn_model_path, sensor_model_path)`. |
| **Factory Pattern** | Instantiates the correct model type based on configuration strings. | `SensorFireRiskModel._create_model()`. |
| **Template Method** | Defines sequential model evaluation loops while letting subroutines plot custom graphs. | `ModelEvaluator` class. |
| **Decorator Pattern** | Validates session authorization prior to executing routes. | `@require_role` decorator. |

---

## 6. Diagnostic Evaluation Suite (`evaluate_models.py`)

Generates statistical validation metrics and outputs comparative plots to `models/plots/`:
*   `confusion_matrices.png`: Computes matrices of prediction errors.
*   `roc_curves.png`: Mapped curve displaying True Positive Rate vs False Positive Rate.
*   `precision_recall_curves.png`: Evaluates model precision limits across varying threshold balances.
*   `metrics_comparison.png`: Combined bar charts comparing model Accuracy, Precision, Recall, and F1 scores.

---

## 7. End-to-End Functional Connection & Data Flow Loops

This section traces how variables, functions, SQL operations, and microcontroller logic outputs interact sequentially.

### Loop A: Wired Serial Telemetry, Inference, and Physical Warning Feedback
```
[Microcontroller Hardware] ---> (Serial Stream) ---> [arduino_reader.py] ---> [NepalFirePredictor] ---> [Database SQL] ---> (Alert check) ---> [arduino_reader.py] ---> (Serial Risk Float) ---> [Microcontroller Hardware]
```
1.  **Hardware Loop**: The microcontroller reads measurements from the physical DHT11 sensor. It formats the data string as `"T:32.4,H:28.1\n"` and outputs it over the serial connection (`COM3` or `/dev/ttyUSB0`) via USB at a rate of 9600 baud.
2.  **Telemetry Polling**: The `ArduinoDataSource.read()` function inside [arduino_reader.py](file:///c:/Users/DELL/Music/fyp/forest-fire-detection/arduino_reader.py) intercepts the raw string, decodes it, and parses out variables using regular expressions (`T:(\d+\.?\d*),H:(\d+\.?\d*)`). It outputs a Python dictionary: `{'temperature': 32.4, 'humidity': 28.1}`.
3.  **Core Prediction Call**: The parsed values are passed to `NepalFirePredictor.predict(32.4, 28.1)`. 
4.  **Feature Processing & Inference**:
    *   Generates engineered features (Vapor Pressure Deficit and Temperature-Humidity Interaction Index).
    *   Feeds these features to the Scikit-learn Logistic Regression model weights.
    *   Evaluates the physical science thresholds ($f(T)$ and $f(\text{RH})$) to compile `Score_science`.
    *   Executes the hybrid calculations: `Risk = (Score_science * 0.8) + (P_Logistic * 0.2)`. It returns a dictionary: `{'fire_risk': 1, 'probability': 0.662, 'risk_level': 'HIGH', ...}`.
5.  **Database Storage**: The helper script invokes `db_mgr.add_sensor_reading(32.4, 28.1, 'HIGH', 0.662)`. This triggers an SQL `INSERT INTO sensor_readings` command.
6.  **Alarms Checking**: If the returned `risk_level` is `HIGH` or `CRITICAL`, the system invokes `db_mgr.add_alert('sensor', reading_id, 'HIGH', '...')`, triggering an SQL `INSERT INTO alerts` with `status='ACTIVE'`.
7.  **Actuation Back-channel**: [arduino_reader.py](file:///c:/Users/DELL/Music/fyp/forest-fire-detection/arduino_reader.py) serializes the probability float as a raw byte string (`"0.66\n"`) and transmits it back over the USB connection.
8.  **Microcontroller LED Actuation**: The hardware parses the float. If the value is $\ge 0.50$, it calls `digitalWrite(RED_LED, HIGH)` and `digitalWrite(BLUE_LED, LOW)`. Otherwise, the Blue LED is turned on.

---

### Loop B: Dashboard Authentication, Session Guarding, and Role Rendering
```
[dashboard.html UI] ---> (Auth Credentials) ---> [app.py /api/auth/login] ---> [Database check] ---> (Session storage) ---> [@require_role Filter] ---> [UI Component Toggle]
```
1.  **Credential Submission**: The operator inputs credentials into the modal layout in [dashboard.html](file:///c:/Users/DELL/Music/fyp/forest-fire-detection/templates/dashboard.html). Clicking Submit invokes `submitLogin()`, which sends a `POST` request to `/api/auth/login` containing `{"username": "admin", "password": "..."}`.
2.  **Secure Hashing Validation**:
    *   `app.py` calls `db_mgr.get_user_by_username("admin")` (queries `users` table).
    *   Validates the incoming plain-text password against the hashed store using Werkzeug's `check_password_hash()`.
3.  **Session Spawning**: If verification succeeds, Flask initializes session variables: `session['username'] = username` and `session['role'] = role`. It returns `{"success": true}`.
4.  **UI Updates**: The client script triggers `checkAuthStatus()`, which fetches `/api/auth/status`. 
    *   It updates `currentUser` in Javascript memory.
    *   If the user's role is `admin`, it executes:
        `document.getElementById('btn-register-trigger').style.display = 'inline-block';`
        `document.getElementById('user-display').textContent = 'Admin: admin';`
    *   It starts periodic page refreshes for sensor data, alerts, and statistics.
5.  **Access Guard Decorator**: When the administrator registers a new operator via the form, the browser POSTs to `/api/auth/register`. Flask executes the `@require_role(['admin'])` wrapper:
    *   If `session.get('role') != 'admin'`, it intercepts execution and aborts with `403 Forbidden`.
    *   Otherwise, it inserts the new operator credentials into the `users` database table.
6.  **Log Out Scrubbing**: When the user logs out (`POST /api/auth/logout`), the server clears the active session. The browser invokes the `clearDashboardData()` Javascript routine, resetting all table logs, statistics counters, gauges, alerts, and graphs to `--` values to prevent unauthenticated data leaks.

---

### Loop C: Multi-Modal Ensemble Evaluation Flow
```
[dashboard.html UI] ---> (Image Upload + Weather values) ---> [app.py /api/predict-ensemble] ---> [Prediction System] ---> [CNN + LogReg Inference] ---> [Database SQL] ---> [UI Response Display]
```
1.  **Payload Submission**: The operator uploads an image and submits temperature/humidity parameters. The dashboard triggers `submitEnsemble()`, which uploads a multipart form containing `image`, `temperature`, and `humidity` inputs to `/api/predict-ensemble`.
2.  **API Parsing & Auth Guard**:
    *   Flask checks if session `role` is valid (`admin` or `operator`).
    *   Saves the uploaded file to disk and converts temperature/humidity parameters into floats.
3.  **Subsystem Instantiation**: `app.py` passes the arguments to:
    *   `ImagePredictionInput(filepath)`
    *   `SensorPredictionInput(temperature, humidity)`
    These constructors validate data ranges (e.g. humidity between 0-100%, files exist).
4.  **Ensemble Scoring Pipeline**:
    `app.py` calls `prediction_system.combined_prediction(image_input, sensor_input)`.
    *   **Visual inference**: Passes the image tensor through the custom CNN network to calculate `image_confidence`.
    *   **Environmental inference**: Invokes the Nepal-calibrated Logistic Regression ruleset to determine `sensor_confidence`.
    *   **Blended Score**: Computes the weighted ensemble score: `Ensemble = (image_confidence * 0.6) + (sensor_confidence * 0.4)`.
5.  **Record Insertion**: Calls `db_mgr.add_ensemble_prediction(...)`, saving the combined parameters into the `ensemble_predictions` database table.
6.  **Alarms Integration**: If the ensemble score is $\ge 0.50$, it automatically invokes `db_mgr.add_alert('ensemble', ensemble_id, 'CRITICAL', '...')`.
7.  **Response Return**: Returns a JSON payload to the browser containing the final classification status, component scores, and emergency recommendations. The dashboard dynamically updates the risk gauges, appends logs to the history table, and issues warning toasts.

---

## 8. Troubleshooting & Resource Guide

### A. Serial Port Locked
```bash
# Check device permissions (Linux)
sudo usermod -a -G dialout $USER
```
On Windows, check device manager to make sure port configuration in `.env` matches the assigned index (e.g., `COM3`).

### B. Flask Server Port Conflict
If port `5000` is bound:
```bash
# Windows
Stop-Service -Name "Spooler"  # Or locate process
# Alternative: Run app on a custom port
$env:FLASK_PORT="5001" ; python app.py
```

### C. System Resource footprint
*   **Runtime Memory Usage**: ~850MB RAM (CNN loading and PyTorch libraries require the majority of this allocation; sensor model requires < 50MB).
*   **Storage Consumption**: Weights require ~500KB total.
