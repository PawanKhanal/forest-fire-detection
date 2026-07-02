"""Flask web dashboard for Forest Fire Detection System."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, send_from_directory, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from train_sensor_model import NepalFirePredictor
from src.inference.predictor import (
    FirePredictionSystem,
    ImagePredictionInput,
    SensorPredictionInput
)
from src.data.database import DatabaseManager

# Load environment variables
load_dotenv()

# Initialize Database Manager
db_mgr = DatabaseManager()

# ---------------------------------------------------------------------------
# Flask Configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.secret_key = os.getenv('SECRET_KEY', 'forest-fire-bca-secret-key-9999')

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
        print('[OK] Image prediction system initialized')
    except Exception as e:
        print(f'[WARN] Image system init failed (image prediction disabled): {e}')
        prediction_system = None  # allow sensor-only operation

    try:
        nepal_predictor = NepalFirePredictor(sensor_model_path)
        print('[OK] Nepal-calibrated sensor predictor initialized')
    except Exception as e:
        print(f'[ERR] Failed to initialize Nepal predictor: {e}')
        raise  # sensor predictor is required


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def require_systems(f):
    """Decorator to ensure all ML systems are ready (for image/ensemble endpoints)."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if prediction_system is None:
            return jsonify({'error': 'Image model not loaded — image prediction unavailable'}), 503
        if nepal_predictor is None:
            return jsonify({'error': 'Sensor model not loaded'}), 503
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Error Handlers (ensure all responses are JSON)
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def handle_404(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def handle_500(e):
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_reading(temperature: float, humidity: float, risk_level: str, probability: float, trigger_alert: bool = True) -> int:
    reading_id = db_mgr.add_sensor_reading(temperature, humidity, risk_level, probability)
    if trigger_alert and risk_level in ['HIGH', 'CRITICAL']:
        db_mgr.add_alert(
            source_type='sensor',
            source_id=reading_id,
            risk_level=risk_level,
            message=f"High fire risk detected by sensors: Temp {temperature:.1f}°C, Humidity {humidity:.1f}%."
        )
    return reading_id


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
@app.route('/api/readings', methods=['GET', 'POST'])
def readings_endpoint() -> Dict[str, Any]:
    """GET: return recent readings. POST: add a new sensor reading."""
    if request.method == 'GET':
        limit = request.args.get('limit', default=20, type=int)
        return jsonify(db_mgr.get_recent_readings(limit))

    # --- POST ---
    if nepal_predictor is None:
        return jsonify({'error': 'Sensor model not initialized'}), 503

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

        # Save to database
        pred_id = db_mgr.add_image_prediction(
            image_path=f"/uploads/{filepath.name}",
            class_name=class_name,
            confidence=result.confidence,
            risk_level=risk_level
        )

        if risk_level in ['HIGH', 'CRITICAL']:
            db_mgr.add_alert(
                source_type='image',
                source_id=pred_id,
                risk_level=risk_level,
                message=f"Image prediction detected {class_name} with {result.confidence:.1%} confidence."
            )

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

        # Save to database
        ens_id = db_mgr.add_ensemble_prediction(
            image_path=f"/uploads/{filepath.name}",
            temperature=temperature,
            humidity=humidity,
            image_confidence=image_result.confidence,
            sensor_confidence=sensor_conf,
            ensemble_confidence=combined,
            risk_level=risk_level
        )
        add_reading(temperature, humidity, risk_level, combined, trigger_alert=False)

        if risk_level in ['HIGH', 'CRITICAL']:
            db_mgr.add_alert(
                source_type='ensemble',
                source_id=ens_id,
                risk_level=risk_level,
                message=f"Ensemble prediction triggered alert: Combined confidence {combined:.1%}, Risk {risk_level}."
            )

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
# Routes - User Authentication
# ---------------------------------------------------------------------------
@app.route('/api/auth/register', methods=['POST'])
def register() -> Dict[str, Any]:
    """Register a new operator/admin user."""
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing username or password'}), 400
        
    username = data['username'].strip()
    password = data['password']
    email = data.get('email', '').strip()
    role = data.get('role', 'operator').strip()
    
    if len(username) < 3 or len(password) < 6:
        return jsonify({'error': 'Username must be >= 3 chars, password >= 6 chars'}), 400
        
    try:
        # Check if user already exists
        if db_mgr.get_user_by_username(username):
            return jsonify({'error': 'Username already exists'}), 400
            
        password_hash = generate_password_hash(password)
        db_mgr.create_user(username, password_hash, email, role)
        return jsonify({'success': True, 'message': 'User registered successfully'})
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login() -> Dict[str, Any]:
    """Login an operator/admin user."""
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing username or password'}), 400
        
    username = data['username'].strip()
    password = data['password']
    
    try:
        user = db_mgr.get_user_by_username(username)
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid username or password'}), 401
            
        session.clear()
        session['username'] = user['username']
        session['role'] = user['role']
        
        return jsonify({
            'success': True,
            'user': {
                'username': user['username'],
                'role': user['role']
            }
        })
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout() -> Dict[str, Any]:
    """Logout current user."""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/auth/status', methods=['GET'])
def auth_status() -> Dict[str, Any]:
    """Get active login session status."""
    if 'username' in session:
        return jsonify({
            'logged_in': True,
            'username': session['username'],
            'role': session.get('role', 'operator')
        })
    return jsonify({'logged_in': False})


# ---------------------------------------------------------------------------
# Routes - System Alerts
# ---------------------------------------------------------------------------
@app.route('/api/alerts', methods=['GET'])
def get_alerts() -> Dict[str, Any]:
    """Get recent alerts list."""
    try:
        limit = request.args.get('limit', default=50, type=int)
        alerts = db_mgr.get_alerts(limit)
        return jsonify(alerts)
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve alerts: {str(e)}'}), 500


@app.route('/api/alerts/resolve', methods=['POST'])
def resolve_alert() -> Dict[str, Any]:
    """Resolve an active alert (authentication required)."""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required to resolve alerts'}), 401
        
    data = request.get_json()
    if not data or 'alert_id' not in data:
        return jsonify({'error': 'Missing alert_id'}), 400
        
    try:
        alert_id = int(data['alert_id'])
        success = db_mgr.resolve_alert(alert_id, session['username'])
        if success:
            return jsonify({'success': True, 'message': f'Alert {alert_id} resolved.'})
        return jsonify({'error': 'Failed to resolve alert'}), 400
    except Exception as e:
        return jsonify({'error': f'Error resolving alert: {str(e)}'}), 500


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
    return jsonify(db_mgr.get_statistics()), 200


# (Error handlers registered above at lines 92-99)


@app.after_request
def after_request(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle CORS preflight requests for all /api/* routes."""
    response = jsonify({})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response, 200


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