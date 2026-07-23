# 🌲 Forest Fire Detection System: Multi-Modal AI & IoT Early Warning System

**BCA Final Year Project | Tribhuvan University**

A production-ready, hybrid artificial intelligence and internet-of-things (IoT) system designed for real-time forest fire detection. The system integrates computer vision (from satellite/aerial imagery) with physical environmental sensors (temperature and humidity) using a multi-modal ensemble model calibrated specifically for the climate of Nepal.

---

## 📌 Table of Contents
1. [System Architecture & Flow](#-system-architecture--flow)
2. [Core Features](#-core-features)
3. [Technology Stack](#-technology-stack)
4. [Algorithms & Mathematical Foundations](#-algorithms--mathematical-foundations)
5. [Database Schema](#-database-schema)
6. [Hardware Layout & Wiring](#-hardware-layout--wiring)
7. [Project Structure](#-project-structure)
8. [Setup & Quick Start](#-setup--quick-start)
9. [Running the System](#-running-the-system)
10. [Model Evaluation & Visualization](#-model-evaluation--visualization)

---

## 🎨 System Architecture & Flow

The system employs an **Interface-Based Object-Oriented Design** following **SOLID principles** to ensure high extensibility (e.g. adding new types of sensors or cloud pipelines without altering core evaluation/prediction logic).

```
                     +----------------------------+
                     |   Physical Forest Environment  |
                     +--------------+-------------+
                                    |
            +-----------------------+-----------------------+
            | (Visual Camera Feed)                          | (DHT11 Sensor Data)
            v                                               v
+-----------------------+                       +-----------------------+
|  Satellite/UAV Image  |                       |  Arduino Uno Board    |
+-----------+-----------+                       +-----------+-----------+
            | (HTTP Upload)                                 | (Serial Port / T:X,H:Y)
            v                                               v
+-----------------------+                       +-----------------------+
|  Flask Web Dashboard  |<=====================>|  python arduino_reader|
+-----------+-----------+                       +-----------+-----------+
            | (ImageInput)                                  | (SensorInput)
            |                                               |
            +-----------------------+-----------------------+
                                    v
                     +----------------------------+
                     |    FirePredictionSystem    |
                     +--------------+-------------+
                                    |
           +------------------------+------------------------+
           | (60% Weight)                                    | (40% Weight)
           v                                                 v
+-----------------------+                       +-----------------------+
|   ResNet18 CNN Model  |                       |  Nepal-Calibrated RF  |
|     (Deep Learning)   |                       |    (Science + ML)     |
+-----------+-----------+                       +-----------+-----------+
            | (Logits / Softmax)                            | (Probability / FWI)
            v                                               v
            +-----------------------+-----------------------+
                                    v
                     +----------------------------+
                     | Ensemble Probability Blend |
                     +--------------+-------------+
                                    |
                                    v
                     +----------------------------+
                     |      PredictionResult      |
                     |  - Fire / No Fire Class    |
                     |  - Risk Level Label        |
                     |  - Safety Recommendation   |
                     +--------------+-------------+
                                    |
            +-----------------------+-----------------------+
            | (Write Database)                              | (Send back via Serial)
            v                                               v
+-----------------------+                       +-----------------------+
|   SQLite / Postgres   |                       |  Arduino LEDs Control |
|   Log & Alert Tables  |                       | (Green/Yellow/Red/Buzz)
+-----------------------+                       +-----------------------+
```

---

## 📈 Core Features

*   **Dual-Sensor Modality (Multi-Modal Ensemble)**: Combines visual information with atmospheric indicators. If a camera lens is blocked by heavy smoke, environmental indicators can still trigger warnings (and vice versa).
*   **Nepal-Calibrated Predictor**: Adjusts standard models using local fire weather principles (combining 80% Canadian Fire Weather Index equations and 20% Portuguese climate-trained Machine Learning predictions). This compensates for global datasets lacking local climate representations.
*   **Interactive Web Dashboard**:
    *   Dynamic Bootstrap 5 dashboard with automatic data refreshing (every 5 seconds).
    *   Dual visual analog gauges for live temperature and humidity.
    *   File drag-n-drop interface for manual visual test uploads.
    *   Integrated REST API allowing programmatic CRUD operations for external nodes.
*   **Continuous Hardware Polling**: Serial monitor client (`arduino_reader.py`) reading live inputs and sending risk indicators back to the physical controller to trigger warning LEDs.
*   **Dual Database Configurations**: Pluggable storage architecture supporting local SQLite for debugging/prototyping and PostgreSQL for enterprise production runs.

---

## 🛠️ Technology Stack

*   **Programming Languages**: Python 3.11, C++ (Arduino sketch)
*   **Machine Learning / Deep Learning**:
    *   `torch` (>=2.0.0) & `torchvision` (>=0.15.0) - Convolutional Neural Networks & Transfer Learning
    *   `scikit-learn` (>=1.3.0) - Random Forest Classifier, Logistic Regression, and metrics pipeline
    *   `joblib` - Model persistence and scaler serialization
*   **Data Wrangling & Analysis**: `pandas` (>=2.0.0), `numpy` (>=1.24.0), `scipy` (>=1.10.0)
*   **Web Framework & Backend Interface**:
    *   `flask` (>=3.0.0) & `werkzeug` (>=3.0.0) - REST APIs & Server routing
    *   `python-dotenv` - Environment parameter management
*   **Visual Assets & Plotting**: `matplotlib` (>=3.7.0), `seaborn` (>=0.12.0), `opencv-python` (>=4.8.0), `pillow` (>=10.0.0)
*   **Hardware Interface**: `pyserial` (>=3.5)
*   **Databases**: `sqlite3` (built-in), `psycopg2-binary` (>=2.9.0) (PostgreSQL interface)
*   **Testing and Linting**: `pytest` (>=7.4.0), `black` (>=23.0.0), `flake8` (>=6.0.0)

---

## 🧮 Algorithms & Mathematical Foundations

### 1. Visual CNN Classifier (ResNet-18 Transfer Learning)
The system adapts the state-of-the-art **ResNet-18** network pre-trained on ImageNet. 
*   **Transfer Learning Strategy**: All pre-trained convolutional features are frozen ($W_{\text{base}}$ parameter updates are skipped) to retain generalized edge/texture detectors.
*   **Custom Classification Head**: The final fully connected layer is replaced with a custom multilayer perceptron (MLP) mapping features to fire status:
    $$\text{Head}(X) = \text{Linear}(128 \to 2)\left(\text{Dropout}_{0.5}\left(\text{ReLU}\left(\text{BatchNorm1d}\left(\text{Linear}(512 \to 128)(X)\right)\right)\right)\right)$$
*   **Objective Function**: Optimized using Cross-Entropy Loss with Adam optimizer:
    $$\mathcal{L} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})$$

---

### 2. Environmental Model Feature Engineering (Random Forest)
The machine learning algorithm takes basic variables (Temperature and Humidity) and constructs higher-order physical predictors to train a **Random Forest Classifier**:

1.  **Vapor Pressure Deficit (VPD)**: Represents the difference between the amount of moisture in the air and how much moisture the air can hold when it is saturated. Higher VPD translates directly to drier vegetation fuels.
    *   **Saturation Vapor Pressure ($e_s$)** in kPa:
        $$e_s = 0.6108 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$$
    *   **Vapor Pressure Deficit (VPD)** in kPa:
        $$\text{VPD} = e_s \times \left(1 - \frac{\text{RH}}{100}\right)$$
2.  **Temperature-Humidity Interaction Index**:
    $$\text{TH}_{\text{interaction}} = \frac{T \times (100 - \text{RH})}{100}$$

---

### 3. Nepal-Calibrated Environmental Predictor
Because generic datasets (like the Portuguese forest fire dataset) are based on Mediterranean environments, standard classifiers can exhibit high false rates in Nepal. The `NepalFirePredictor` solves this by introducing a science-based hybrid calculation:

*   **Science Risk Factors**:
    *   $f(T)$ returns $0.05$ to $0.95$ risk based on temperature step brackets.
    *   $f(\text{RH})$ returns $0.05$ to $0.95$ risk based on humidity dryness brackets.
    *   **Science Score**:
        $$\text{Score}_{\text{science}} = 0.5 \times f(T) + 0.5 \times f(\text{RH})$$
*   **Hybrid Prediction Formula**: Combines the physical science score (80% weight) with the Random Forest probability output (20% weight):
    $$\text{Risk}_{\text{final}} = \left(\text{Score}_{\text{science}} \times 0.8\right) + \left(P_{\text{RF}}(\text{Fire}) \times 0.2\right)$$
*   **Risk Level Categorization**:
    *   $\text{Risk}_{\text{final}} < 0.20 \implies \text{\textbf{LOW}}$
    *   $0.20 \le \text{Risk}_{\text{final}} < 0.50 \implies \text{\textbf{MEDIUM}}$
    *   $0.50 \le \text{Risk}_{\text{final}} < 0.80 \implies \text{\textbf{HIGH}}$
    *   $\text{Risk}_{\text{final}} \ge 0.80 \implies \text{\textbf{CRITICAL}}$

---

### 4. Ensemble Prediction
When both visual feed and hardware telemetry are available, the system aggregates predictions:
$$\text{Confidence}_{\text{Ensemble}} = 0.6 \times \text{Confidence}_{\text{Image}} + 0.4 \times \text{Probability}_{\text{Sensor}}$$

---

## 🗄️ Database Schema

The system supports SQLite and PostgreSQL. Upon launch, `DatabaseManager` runs schema scripts to initialize the following tables:

```
┌────────────────────────────────────────────────────────────────────────┐
│                                 USERS                                  │
├───────────────┬──────────────────────┬─────────────────────────────────┤
│ id            │ SERIAL (PK)          │ Auto-incrementing identifier    │
│ username      │ VARCHAR(50) (UNIQUE) │ Account name                    │
│ password_hash │ VARCHAR(255)         │ Secure salted hash              │
│ email         │ VARCHAR(100)         │ Optional contact mail           │
│ role          │ VARCHAR(20)          │ Access role (operator/admin)    │
│ created_at    │ TIMESTAMP            │ Generation timestamp            │
└───────────────┴──────────────────────┴─────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                            SENSOR_READINGS                             │
├───────────────┬──────────────────────┬─────────────────────────────────┤
│ id            │ SERIAL (PK)          │ Auto-incrementing identifier    │
│ timestamp     │ VARCHAR(30)          │ Date & Time of transmission     │
│ temperature   │ DOUBLE PRECISION     │ Temperature in Celsius          │
│ humidity      │ DOUBLE PRECISION     │ Relative humidity percentage    │
│ risk_level    │ VARCHAR(20)          │ Classified risk tag             │
│ probability   │ DOUBLE PRECISION     │ Calculated probability (0-1)    │
└───────────────┴──────────────────────┴─────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                            IMAGE_PREDICTIONS                           │
├───────────────┬──────────────────────┬─────────────────────────────────┤
│ id            │ SERIAL (PK)          │ Auto-incrementing identifier    │
│ timestamp     │ VARCHAR(30)          │ Processing timestamp            │
│ image_path    │ VARCHAR(255)         │ Upload location                 │
│ class_name    │ VARCHAR(50)          │ FIRE or NO_FIRE                 │
│ confidence    │ DOUBLE PRECISION     │ Probability value               │
│ risk_level    │ VARCHAR(20)          │ Classified risk tag             │
└───────────────┴──────────────────────┴─────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                          ENSEMBLE_PREDICTIONS                          │
├─────────────────────┬──────────────────┬───────────────────────────────┤
│ id                  │ SERIAL (PK)      │ Auto-incrementing identifier  │
│ timestamp           │ VARCHAR(30)      │ Evaluation timestamp          │
│ image_path          │ VARCHAR(255)     │ Image location                │
│ temperature         │ DOUBLE PRECISION │ Celsius measurement           │
│ humidity            │ DOUBLE PRECISION │ Humidity measurement          │
│ image_confidence    │ DOUBLE PRECISION │ Image model score             │
│ sensor_confidence   │ DOUBLE PRECISION │ Sensor model score            │
│ ensemble_confidence │ DOUBLE PRECISION │ Combined ensemble score       │
│ risk_level          │ VARCHAR(20)      │ Final risk rating             │
└─────────────────────┴──────────────────┴───────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                                ALERTS                                  │
├───────────────┬──────────────────────┬─────────────────────────────────┤
│ id            │ SERIAL (PK)          │ Auto-incrementing identifier    │
│ timestamp     │ VARCHAR(30)          │ Trigger date & time             │
│ source_type   │ VARCHAR(20)          │ SENSOR, IMAGE, or ENSEMBLE      │
│ source_id     │ INTEGER              │ ID of row in source table       │
│ risk_level    │ VARCHAR(20)          │ Severity level                  │
│ message       │ TEXT                 │ Warning log details             │
│ status        │ VARCHAR(20)          │ ACTIVE or RESOLVED              │
│ resolved_by   │ INTEGER (FK)         │ User ID that closed the alert   │
│ resolved_at   │ VARCHAR(30)          │ Timestamp when resolved         │
└───────────────┴──────────────────────┴─────────────────────────────────┘
```

---

## 🔌 Hardware Layout & Wiring

To deploy the physical module, wire an **Arduino Uno** (or similar ATmega328P microcontroller) with the components as detailed below:

```
            +---------------------------------+
            |           ARDUINO UNO           |
            |                                 |
            |     5V ---[VCC DHT11]           |
            |    GND ---[GND DHT11]           |
            |  Pin 2 ---[DATA DHT11]          |
            |                                 |
            |  Pin 8 ---[Resistor 220Ω]---[+] | Green LED (Low Risk)
            |            GND -------------[-] |
            |                                 |
            |  Pin 9 ---[Resistor 220Ω]---[+] | Yellow LED (Medium Risk)
            |            GND -------------[-] |
            |                                 |
            | Pin 10 ---[Resistor 220Ω]---[+] | Red LED (High/Critical Risk)
            |            GND -------------[-] |
            +---------------------------------+
```

### Components Checklist:
1.  **Arduino Uno** / Nano / Mega
2.  **DHT11** Temperature & Humidity Sensor
3.  **3x LEDs** (1 Green, 1 Yellow, 1 Red)
4.  **3x 220 Ohm Resistors** (to limit LED current)
5.  **Breadboard & Jumper wires**
6.  **USB Cable** (A-to-B for PC serial communication)

---

## 📁 Project Structure

```
forest-fire-detection/
│
├── arduino/
│   ├── arduino_dht11_fire.ino    # Arduino sketch to read DHT11 and control LEDs (Serial)
│   └── esp_dht11_restapi.ino     # ESP8266/ESP32 sketch to read DHT11 and post via HTTP REST API
│
├── config/
│   └── config.yaml               # YAML global configuration file
│
├── data/
│   ├── raw/                      # Downloaded image folders (train, val, test)
│   ├── sensor/
│   │   ├── forestfires.csv       # Historical weather wildfire data (UCI)
│   │   └── readings_history.json # Local backup history logs
│   └── forest_fire.db            # Default SQLite Database file
│
├── models/
│   ├── plots/                    # Evaluation performance visualizations
│   ├── saved/
│   │   ├── forest_fire_cnn_final.pth # Serialized PyTorch CNN model weights
│   │   └── sensor_model.pkl      # Serialized Scikit-Learn Random Forest model
│   ├── cnn_model_pytorch.py      # ResNet18 CNN model training classes
│   └── sensor_model.py           # Feature engineering & Random Forest classes
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Image file utilities
│   │   ├── preprocessing.py      # Standardizing and resizing pipeline
│   │   ├── pytorch_dataset.py    # Torch Dataset wrapping
│   │   └── database.py           # Pluggable SQL database manager
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py          # Unified predictor API (Strategy Pattern)
│
├── templates/
│   └── dashboard.html            # Bootstrap 5 real-time HTML UI dashboard
│
├── tests/                        # PyTest regression testing scripts
│   └── test_standardization.py
│
├── uploads/                      # Temporary repository for web uploads
├── logs/                         # File logging directory
│
├── .env.example                  # Environment variable configuration template
├── .gitignore                    # Version control ignore lists
├── app.py                        # Web server (Flask application)
├── arduino_reader.py             # Python serial connection monitor daemon
├── download_fire_dataset.py      # Dataset visual importer (Kaggle hub)
├── download_portuguese_fire_data.py # UCI dataset importer (urllib)
├── evaluate_models.py            # Diagnostic & performance evaluation suite
├── examples.py                   # Sandbox code demonstration script
├── requirements.txt              # Project package declarations
├── setup.py                      # Local packages metadata installer
└── README.md                     # Documentation file (This File)
```

---

## 🚀 Setup & Quick Start

### 1. Clone the Project & Navigate
```bash
git clone https://github.com/PawanKhanal/forest-fire-detection.git
cd forest-fire-detection
```

### 2. Configure Python Virtual Environment
It is highly recommended to run this project inside a virtual environment to prevent package version conflicts:
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create and Configure Environment Settings
Copy the `.env.example` file to `.env` and fill in the paths and preferences:
```bash
cp .env.example .env
```
*Open `.env` in a text editor to verify local paths, serial port address, and database settings (default is SQLite).*

---

## 🏃 Running the System

### Phase A: Acquire Datasets and Train Models
If you are running the system for the first time, you must download the training data and save the model weights.

```bash
# 1. Download Kaggle images and UCI weather data
python download_fire_dataset.py
python download_portuguese_fire_data.py

# 2. Train the CNN (PyTorch)
python train_pytorch.py

# 3. Train the Sensor Model (Random Forest)
python train_sensor_model.py
```
*Note: Make sure your trained models are saved in the `models/saved/` folder as `forest_fire_cnn_final.pth` and `sensor_model.pkl`.*

---

### Phase B: Launching Hardware Integration

#### Option 1: Wired Serial Connection (Arduino Uno)
1.  Connect your Arduino board to the PC via USB cable.
2.  Open the Arduino IDE, load [arduino_dht11_fire.ino](file:///c:/Users/DELL/Music/fyp/forest-fire-detection/arduino/arduino_dht11_fire.ino), choose the appropriate board/port, and click **Upload**.
3.  Modify `ARDUINO_PORT` in your `.env` file to match the assigned port (e.g. `COM3` on Windows, or `/dev/ttyUSB0` on Linux).
4.  Run the Python listener daemon:
    ```bash
    python arduino_reader.py
    ```
    *The console will display color-coded risk levels (🟢 Green / 🟡 Yellow / 🔴 Red) as data streams in.*

#### Option 2: Wireless IoT Connection (ESP8266 / ESP32 via HTTP REST API)
1.  Open the Arduino IDE.
2.  Install the required libraries:
    *   **DHT sensor library** (by Adafruit)
    *   **Adafruit Unified Sensor** (by Adafruit)
    *   **ArduinoJson** (by Benoit Blanchon, v6 or v7)
3.  Load [esp_dht11_restapi.ino](file:///c:/Users/DELL/Music/fyp/forest-fire-detection/arduino/esp_dht11_restapi.ino) into the IDE.
4.  Configure your Wi-Fi details and the IP address of your Flask server:
    ```cpp
    const char* ssid = "YOUR_WIFI_SSID";
    const char* password = "YOUR_WIFI_PASSWORD";
    const char* serverUrl = "http://<YOUR_FLASK_SERVER_IP>:5000/api/readings";
    ```
5.  Select your ESP8266 or ESP32 board and upload the code.
6.  Start your Flask application (`python app.py`).
7.  The ESP module will automatically connect to Wi-Fi, query the DHT11, and perform non-blocking HTTP POST requests. The risk analysis will run on the server, log into the SQL database, and update the physical LEDs in real-time. No background python scripts needed on the PC!

---

### Phase C: Running the Web Dashboard
Open a separate terminal window, activate the virtual environment, and boot the web application server:

```bash
python app.py
```
By default, the server spins up at: **`http://localhost:5000`**

#### Key Dashboard Functions:
*   **Sensor Gauges**: Displays live temperature and humidity measurements reported by the hardware module.
*   **Image Detection**: Upload custom photos to check for visual signs of fire.
*   **Ensemble Evaluation**: Submit both variables simultaneously to run the weighted ensemble model.

---

## 📊 Model Evaluation & Visualization

To inspect validation metrics, generate confusion matrices, ROC/AUC indicators, and compare model precision/recall performance metrics, run:

```bash
python evaluate_models.py
```

This script evaluates both models against test splits and outputs the diagnostic visual assets to `models/plots/`:
*   `confusion_matrices.png` - Normal and normalized confusion matrices.
*   `roc_curves.png` - True positive vs. false positive rate mappings.
*   `precision_recall_curves.png` - Model behavior under different class balances.
*   `metrics_comparison.png` - Comparative bar chart mapping Accuracy, F1-Score, Precision, and Recall.
*   `evaluation_report.txt` - Text report containing classification matrices and parameters.
