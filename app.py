"""Flask web dashboard for Forest Fire Detection System."""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from src.inference.predictor import (
    FirePredictionSystem,
    ImagePredictionInput,
    SensorPredictionInput
)

# Load environment variables
load_dotenv()

# Flask configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# Global prediction system
prediction_system: FirePredictionSystem = None

# In-memory readings storage (for demo; use database in production)
readings_store: List[Dict[str, Any]] = []


def init_prediction_system() -> None:
    """Initialize the prediction system on app startup."""
    global prediction_system
    
    try:
        cnn_model_path = os.getenv(
            'CNN_MODEL_PATH',
            'models/saved/forest_fire_cnn_final.pth'
        )
        sensor_model_path = os.getenv(
            'SENSOR_MODEL_PATH',
            'models/saved/sensor_model.pkl'
        )
        
        prediction_system = FirePredictionSystem(
            cnn_model_path=cnn_model_path,
            sensor_model_path=sensor_model_path,
            device='cpu'
        )
        print("✓ Prediction system initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize prediction system: {str(e)}")
        raise


def require_prediction_system(f):
    """Decorator to check if prediction system is initialized."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if prediction_system is None:
            return jsonify({'error': 'Prediction system not initialized'}), 503
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_reading(
    temperature: float,
    humidity: float,
    risk_level: str,
    probability: float
) -> None:
    """Add a reading to the store."""
    readings_store.append({
        'timestamp': datetime.now().isoformat(),
        'temperature': temperature,
        'humidity': humidity,
        'risk_level': risk_level,
        'probability': probability
    })
    
    # Keep only last 100 readings
    if len(readings_store) > 100:
        readings_store.pop(0)


@app.route('/')
def dashboard() -> str:
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/readings', methods=['GET'])
def get_readings() -> Dict[str, Any]:
    """Get recent sensor readings."""
    limit = request.args.get('limit', default=20, type=int)
    return jsonify(readings_store[-limit:])


@app.route('/api/readings', methods=['POST'])
@require_prediction_system
def add_sensor_reading() -> Dict[str, Any]:
    """Add a new sensor reading via API."""
    data = request.get_json()
    
    if not data or 'temperature' not in data or 'humidity' not in data:
        return jsonify({'error': 'Missing temperature or humidity'}), 400
    
    try:
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        
        sensor_input = SensorPredictionInput(temperature, humidity)
        result = prediction_system.predict_from_sensors(sensor_input)
        
        add_reading(
            temperature=temperature,
            humidity=humidity,
            risk_level=result.risk_level,
            probability=result.confidence
        )
        
        return jsonify({
            'success': True,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'risk_level': result.risk_level,
            'temperature': temperature,
            'humidity': humidity
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict-image', methods=['POST'])
@require_prediction_system
def predict_image() -> Dict[str, Any]:
    """Predict fire presence from uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = app.config['UPLOAD_FOLDER'] / f"{timestamp}{filename}"
        file.save(filepath)
        
        # Make prediction
        image_input = ImagePredictionInput(str(filepath))
        result = prediction_system.predict_from_image(image_input)
        
        return jsonify({
            'success': True,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'risk_level': result.risk_level,
            'class_name': result.metadata['class_name'],
            'image_path': f"/uploads/{timestamp}{filename}"
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict-ensemble', methods=['POST'])
@require_prediction_system
def predict_ensemble() -> Dict[str, Any]:
    """Combined image and sensor prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    data = request.form
    if 'temperature' not in data or 'humidity' not in data:
        return jsonify({'error': 'Missing temperature or humidity'}), 400
    
    try:
        # Process image
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = app.config['UPLOAD_FOLDER'] / f"{timestamp}{filename}"
        file.save(filepath)
        
        # Prepare inputs
        image_input = ImagePredictionInput(str(filepath))
        sensor_input = SensorPredictionInput(
            float(data['temperature']),
            float(data['humidity'])
        )
        
        # Get ensemble prediction
        result = prediction_system.combined_prediction(
            image_input=image_input,
            sensor_input=sensor_input,
            image_weight=0.6,
            sensor_weight=0.4
        )
        
        add_reading(
            temperature=sensor_input.temperature,
            humidity=sensor_input.humidity,
            risk_level=result.risk_level,
            probability=result.confidence
        )
        
        return jsonify({
            'success': True,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'risk_level': result.risk_level,
            'image_path': f"/uploads/{timestamp}{filename}",
            'metadata': result.metadata
        })
    
    except Exception as e:
        return jsonify({'error': f'Ensemble prediction failed: {str(e)}'}), 500


@app.route('/api/model-info', methods=['GET'])
@require_prediction_system
def get_model_info() -> Dict[str, Any]:
    """Get information about loaded models."""
    return jsonify(prediction_system.get_model_info())


@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename: str) -> Any:
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/statistics', methods=['GET'])
def get_statistics() -> Dict[str, Any]:
    """Get statistics from readings."""
    if not readings_store:
        return jsonify({'error': 'No readings available'}), 400
    
    temperatures = [r['temperature'] for r in readings_store]
    humidities = [r['humidity'] for r in readings_store]
    probabilities = [r['probability'] for r in readings_store]
    
    return jsonify({
        'total_readings': len(readings_store),
        'temperature': {
            'min': min(temperatures),
            'max': max(temperatures),
            'avg': sum(temperatures) / len(temperatures)
        },
        'humidity': {
            'min': min(humidities),
            'max': max(humidities),
            'avg': sum(humidities) / len(humidities)
        },
        'probability': {
            'min': min(probabilities),
            'max': max(probabilities),
            'avg': sum(probabilities) / len(probabilities)
        }
    })


@app.errorhandler(404)
def not_found(error: Any) -> tuple:
    """Handle 404 errors."""
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def server_error(error: Any) -> tuple:
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


@app.before_request
def before_request() -> None:
    """Execute before each request."""
    pass


@app.after_request
def after_request(response: Any) -> Any:
    """Execute after each request."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    init_prediction_system()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv('FLASK_DEBUG', 'False') == 'True'
    )
