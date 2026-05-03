"""Flask web dashboard for Forest Fire Detection System."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from train_sensor_model import NepalFirePredictor
from src.inference.predictor import (
    FirePredictionSystem,
    ImagePredictionInput,
    SensorPredictionInput
)

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Flask Configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
prediction_system: FirePredictionSystem = None
nepal_predictor: NepalFirePredictor = None
readings_store: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_prediction_system() -> None:
    """Initialize both prediction systems on app startup."""
    global prediction_system, nepal_predictor

    cnn_model_path = os.getenv('CNN_MODEL_PATH', 'models/saved/forest_fire_cnn_final.pth')
    sensor_model_path = os.getenv('SENSOR_MODEL_PATH', 'models/saved/sensor_model.pkl')

    try:
        prediction_system = FirePredictionSystem(
            cnn_model_path=cnn_model_path,
            sensor_model_path=sensor_model_path,
            device='cpu'
        )
        prediction_system.class_names = ['FIRE', 'NO_FIRE']
        print("✓ Image prediction system initialized")
    except Exception as e:
        print(f"✗ Failed to initialize image system: {e}")
        raise

    try:
        nepal_predictor = NepalFirePredictor(sensor_model_path)
        print("✓ Nepal-calibrated sensor predictor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Nepal predictor: {e}")
        raise


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def require_systems(f):
    """Decorator to ensure all systems are ready."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if prediction_system is None or nepal_predictor is None:
            return jsonify({'error': 'Systems not initialized'}), 503
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_reading(temperature: float, humidity: float, risk_level: str, probability: float) -> None:
    readings_store.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'temperature': temperature,
        'humidity': humidity,
        'risk_level': risk_level,
        'probability': probability
    })
    if len(readings_store) > 100:
        readings_store.pop(0)


def image_risk_level(class_name: str, confidence: float) -> str:
    """Determine risk level from image prediction."""
    if class_name == 'FIRE':
        if confidence < 0.5:
            return "MEDIUM"
        elif confidence < 0.75:
            return "HIGH"
        return "CRITICAL"
    else:
        return "LOW" if confidence > 0.80 else "MEDIUM"


def save_upload(file) -> Path:
    """Save uploaded file and return its path."""
    filename = secure_filename(file.filename or "image.jpg")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filepath = app.config['UPLOAD_FOLDER'] / f"{timestamp}{filename}"
    file.save(filepath)
    return filepath


# ---------------------------------------------------------------------------
# Routes - Pages
# ---------------------------------------------------------------------------
@app.route('/')
def dashboard() -> str:
    return render_template('dashboard.html')


# ---------------------------------------------------------------------------
# Routes - Sensor Readings (Nepal-Calibrated)
# ---------------------------------------------------------------------------
@app.route('/api/readings', methods=['GET'])
def get_readings() -> Dict[str, Any]:
    limit = request.args.get('limit', default=20, type=int)
    return jsonify(readings_store[-limit:])


@app.route('/api/readings', methods=['POST'])
@require_systems
def add_sensor_reading() -> Dict[str, Any]:
    """Add sensor reading using Nepal-calibrated predictor."""
    data = request.get_json()
    if not data or 'temperature' not in data or 'humidity' not in data:
        return jsonify({'error': 'Missing temperature or humidity'}), 400

    try:
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])

        # Use Nepal-calibrated predictor (correct for our climate)
        result = nepal_predictor.predict(temperature, humidity)

        add_reading(temperature, humidity, result['risk_level'], result['probability'])

        return jsonify({
            'success': True,
            'prediction': int(result['fire_risk']),
            'confidence': float(result['probability']),
            'risk_level': str(result['risk_level']),
            'temperature': float(temperature),
            'humidity': float(humidity),
            'recommendation': str(result['recommendation'])
        })

    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


# ---------------------------------------------------------------------------
# Routes - Image Prediction
# ---------------------------------------------------------------------------
@app.route('/api/predict-image', methods=['POST'])
@require_systems
def predict_image() -> Dict[str, Any]:
    """Predict fire from uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or missing file'}), 400

    try:
        filepath = save_upload(file)
        image_input = ImagePredictionInput(str(filepath))
        result = prediction_system.predict_from_image(image_input)

        class_name = result.metadata.get('class_name', 'UNKNOWN')
        risk_level = image_risk_level(class_name, result.confidence)

        return jsonify({
            'success': True,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'risk_level': risk_level,
            'class_name': class_name,
            'image_path': f"/uploads/{filepath.name}"
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


# ---------------------------------------------------------------------------
# Routes - Ensemble Prediction
# ---------------------------------------------------------------------------
@app.route('/api/predict-ensemble', methods=['POST'])
@require_systems
def predict_ensemble() -> Dict[str, Any]:
    """Combined image + Nepal-calibrated sensor prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    data = request.form
    if 'temperature' not in data or 'humidity' not in data:
        return jsonify({'error': 'Missing temperature or humidity'}), 400

    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or missing file'}), 400

    try:
        filepath = save_upload(file)
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])

        # Use Nepal predictor for sensor component
        sensor_result = nepal_predictor.predict(temperature, humidity)
        image_input = ImagePredictionInput(str(filepath))
        image_result = prediction_system.predict_from_image(image_input)

        # Weighted ensemble
        image_conf = image_result.confidence if image_result.metadata.get('class_name') == 'FIRE' else (1 - image_result.confidence)
        sensor_conf = sensor_result['probability']
        combined = 0.6 * image_conf + 0.4 * sensor_conf

        # Determine ensemble risk level
        if combined < 0.25:
            risk_level = "LOW"
        elif combined < 0.50:
            risk_level = "MEDIUM"
        elif combined < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        add_reading(temperature, humidity, risk_level, combined)

        return jsonify({
            'success': True,
            'prediction': combined >= 0.5,
            'confidence': combined,
            'risk_level': risk_level,
            'image_path': f"/uploads/{filepath.name}",
            'image_class': image_result.metadata.get('class_name'),
            'image_confidence': image_result.confidence,
            'sensor_confidence': sensor_conf
        })

    except Exception as e:
        return jsonify({'error': f'Ensemble prediction failed: {e}'}), 500


# ---------------------------------------------------------------------------
# Routes - Info & Static
# ---------------------------------------------------------------------------
@app.route('/api/model-info', methods=['GET'])
@require_systems
def get_model_info() -> Dict[str, Any]:
    return jsonify(prediction_system.get_model_info())


@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_upload_file(filename: str):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/statistics', methods=['GET'])
def get_statistics() -> Dict[str, Any]:
    if not readings_store:
        return jsonify({'error': 'No readings available'}), 400

    temps = [r['temperature'] for r in readings_store]
    hums = [r['humidity'] for r in readings_store]
    probs = [r['probability'] for r in readings_store]

    return jsonify({
        'total_readings': len(readings_store),
        'temperature': {'min': min(temps), 'max': max(temps), 'avg': round(sum(temps)/len(temps), 1)},
        'humidity': {'min': min(hums), 'max': max(hums), 'avg': round(sum(hums)/len(hums), 1)},
        'probability': {'min': min(probs), 'max': max(probs), 'avg': round(sum(probs)/len(probs), 3)}
    })


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.after_request
def after_request(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    init_prediction_system()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv('FLASK_DEBUG', 'False') == 'True'
    )