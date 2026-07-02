"""Unit tests for label standardization and database manager."""

import pytest
import os
import sqlite3
from unittest.mock import MagicMock, patch
import torch
import numpy as np

from src.inference.predictor import FirePredictionSystem, ImagePredictionInput
from src.data.database import DatabaseManager

def test_database_manager():
    """Test DatabaseManager initialization and insert/query operations."""
    db_path = "data/test_forest_fire.db"
    
    # Clean up previous test database
    if os.path.exists(db_path):
        os.remove(db_path)
        
    try:
        db_mgr = DatabaseManager(db_path)
        
        # Test add_sensor_reading
        read_id = db_mgr.add_sensor_reading(
            temperature=35.5,
            humidity=15.0,
            risk_level="HIGH",
            probability=0.72
        )
        assert read_id == 1
        
        # Test get_recent_readings
        readings = db_mgr.get_recent_readings(1)
        assert len(readings) == 1
        assert readings[0]['temperature'] == 35.5
        assert readings[0]['humidity'] == 15.0
        assert readings[0]['risk_level'] == "HIGH"
        assert readings[0]['probability'] == 0.72
        
        # Test add_image_prediction
        img_id = db_mgr.add_image_prediction(
            image_path="/uploads/test.jpg",
            class_name="FIRE",
            confidence=0.91,
            risk_level="CRITICAL"
        )
        assert img_id == 1
        
        # Test add_ensemble_prediction
        ens_id = db_mgr.add_ensemble_prediction(
            image_path="/uploads/test.jpg",
            temperature=32.0,
            humidity=25.0,
            image_confidence=0.91,
            sensor_confidence=0.65,
            ensemble_confidence=0.81,
            risk_level="CRITICAL"
        )
        assert ens_id == 1
        
        # Test get_statistics
        stats = db_mgr.get_statistics()
        assert stats['total_readings'] == 1
        assert stats['temperature']['min'] == 35.5
        assert stats['temperature']['max'] == 35.5
        
    finally:
        # Clean up database
        if os.path.exists(db_path):
            os.remove(db_path)

@patch('src.inference.predictor.ForestFireCNN')
@patch('src.inference.predictor.SensorFireRiskModel')
def test_label_standardization(mock_sensor_cls, mock_cnn_cls):
    """Test that cnn prediction values are standardized (0: NO_FIRE, 1: FIRE)."""
    # Set up mocks
    mock_cnn = MagicMock()
    mock_cnn_cls.load.return_value = mock_cnn
    
    mock_sensor = MagicMock()
    mock_sensor_cls.load.return_value = mock_sensor
    
    # Initialize prediction system
    system = FirePredictionSystem(
        cnn_model_path="models/saved/forest_fire_cnn_final.pth",
        sensor_model_path="models/saved/sensor_model.pkl",
        device="cpu"
    )
    
    # Test case 1: Model outputs logits where index 0 is largest (raw index 0 = fire)
    # This should be mapped to standardized prediction 1 (FIRE)
    mock_cnn.return_value = torch.tensor([[2.0, -1.0]]) # index 0 is max
    
    # Mock image loading
    with patch.object(ImagePredictionInput, 'validate', return_value=True), \
         patch.object(ImagePredictionInput, 'load_image', return_value=torch.zeros((1, 3, 224, 224))):
        
        result = system.predict_from_image(ImagePredictionInput("test_image.jpg"))
        
        assert result.prediction == 1 # Standardized index 1 is FIRE
        assert result.metadata['class_name'] == "FIRE"
        assert result.risk_level in ["MEDIUM", "HIGH", "CRITICAL"] # Since prediction is 1
        
    # Test case 2: Model outputs logits where index 1 is largest (raw index 1 = nofire)
    # This should be mapped to standardized prediction 0 (NO_FIRE)
    mock_cnn.return_value = torch.tensor([[-1.0, 2.0]]) # index 1 is max
    
    with patch.object(ImagePredictionInput, 'validate', return_value=True), \
         patch.object(ImagePredictionInput, 'load_image', return_value=torch.zeros((1, 3, 224, 224))):
        
        result = system.predict_from_image(ImagePredictionInput("test_image.jpg"))
        
        assert result.prediction == 0 # Standardized index 0 is NO_FIRE
        assert result.metadata['class_name'] == "NO_FIRE"
        assert result.risk_level == "LOW" # Since prediction is 0

def test_user_and_alerts_operations():
    """Test user registration, login, alerts creation, and alerts resolution."""
    db_path = "data/test_auth_alerts.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
        
    try:
        db_mgr = DatabaseManager(db_path)
        
        # Test User Creation
        user_id = db_mgr.create_user(
            username="testop",
            password_hash="hashedpassword123",
            email="testop@example.com",
            role="operator"
        )
        assert user_id == 1
        
        # Test Retrieve User
        user = db_mgr.get_user_by_username("testop")
        assert user is not None
        assert user['username'] == "testop"
        assert user['email'] == "testop@example.com"
        assert user['role'] == "operator"
        
        # Test Retrieve Non-existent User
        non_user = db_mgr.get_user_by_username("nonexistent")
        assert non_user is None
        
        # Test Add Alert
        alert_id = db_mgr.add_alert(
            source_type="sensor",
            source_id=10,
            risk_level="CRITICAL",
            message="Critical sensor risk detected"
        )
        assert alert_id == 1
        
        # Test Get Alerts
        alerts = db_mgr.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]['risk_level'] == "CRITICAL"
        assert alerts[0]['status'] == "ACTIVE"
        assert alerts[0]['resolved_by'] is None
        
        # Test Resolve Alert
        success = db_mgr.resolve_alert(alert_id, "testop")
        assert success is True
        
        # Test Get Alerts after resolution
        alerts_after = db_mgr.get_alerts()
        assert len(alerts_after) == 1
        assert alerts_after[0]['status'] == "RESOLVED"
        assert alerts_after[0]['resolved_by'] == "testop"
        assert alerts_after[0]['resolved_at'] is not None
        
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
