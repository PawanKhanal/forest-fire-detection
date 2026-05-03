"""Arduino serial reader for real-time fire risk monitoring."""

import sys
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
import serial
from pathlib import Path

from models.sensor_model import SensorFireRiskModel
from src.inference.predictor import FirePredictionSystem, SensorPredictionInput


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def read(self) -> Optional[Dict[str, float]]:
        """Read data from source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass


class ArduinoDataSource(DataSource):
    """Serial data reader for Arduino sensors."""
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        timeout: float = 1.0
    ):
        """
        Initialize Arduino data source.
        
        Args:
            port: Serial port device path
            baudrate: Serial communication speed
            timeout: Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None
    
    def connect(self) -> bool:
        """
        Establish serial connection to Arduino.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for Arduino initialization
            print(f"✓ Connected to Arduino on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"✗ Failed to connect to Arduino: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error during connection: {str(e)}")
            return False
    
    def read(self) -> Optional[Dict[str, float]]:
        """
        Read temperature and humidity from Arduino.
        
        Expected format: "T:25.5,H:45.2"
        
        Returns:
            Dictionary with 'temperature' and 'humidity' keys, or None if read fails
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("✗ Serial connection not open")
            return None
        
        try:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                return self._parse_arduino_data(line)
        except UnicodeDecodeError as e:
            print(f"✗ Encoding error: {str(e)}")
        except Exception as e:
            print(f"✗ Read error: {str(e)}")
        
        return None
    
    def disconnect(self) -> None:
        """Close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("✓ Disconnected from Arduino")
    
    @staticmethod
    def _parse_arduino_data(data_string: str) -> Optional[Dict[str, float]]:
        """
        Parse Arduino sensor data.
        
        Args:
            data_string: Raw data string from Arduino
            
        Returns:
            Dictionary with temperature and humidity, or None if parsing fails
        """
        try:
            parts = data_string.split(',')
            if len(parts) < 2:
                return None
            
            temp_str = parts[0].split(':')[1].strip()
            humidity_str = parts[1].split(':')[1].strip()
            
            return {
                'temperature': float(temp_str),
                'humidity': float(humidity_str)
            }
        except (IndexError, ValueError, AttributeError):
            return None


class FireRiskMonitor:
    """Real-time fire risk monitoring system."""
    
    def __init__(
        self,
        data_source: DataSource,
        sensor_model_path: str,
        update_interval: float = 5.0
    ):
        """
        Initialize fire risk monitor.
        
        Args:
            data_source: Data source implementation (e.g., ArduinoDataSource)
            sensor_model_path: Path to trained sensor model
            update_interval: Time between readings in seconds
        """
        self.data_source = data_source
        self.update_interval = update_interval
        self.running = False
        self.readings_history: list = []
        
        try:
            self.sensor_model = SensorFireRiskModel.load(sensor_model_path)
        except Exception as e:
            print(f"✗ Failed to load sensor model: {str(e)}")
            raise
    
    def start(self) -> None:
        """Start monitoring loop."""
        if not self.data_source.connect():
            print("✗ Failed to connect to data source")
            return
        
        self.running = True
        print("\n" + "="*70)
        print("FOREST FIRE DETECTION SYSTEM - REAL-TIME MONITORING")
        print("="*70 + "\n")
        
        try:
            while self.running:
                self._process_reading()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\n\n✓ Monitoring stopped by user")
        except Exception as e:
            print(f"\n✗ Monitor error: {str(e)}")
        finally:
            self.stop()
    
    def _process_reading(self) -> None:
        """Process a single sensor reading."""
        sensor_data = self.data_source.read()
        
        if sensor_data is None:
            return
        
        try:
            sensor_input = SensorPredictionInput(
                temperature=sensor_data['temperature'],
                humidity=sensor_data['humidity']
            )
            
            result = self.sensor_model.predict(
                sensor_data['temperature'],
                sensor_data['humidity']
            )
            
            self._display_reading(sensor_data, result)
            self.readings_history.append({
                'timestamp': datetime.now(),
                'data': sensor_data,
                'result': result
            })
        except Exception as e:
            print(f"✗ Prediction error: {str(e)}")
    
    def _display_reading(
        self,
        sensor_data: Dict[str, float],
        result: Dict[str, Any]
    ) -> None:
        """Display formatted sensor reading and prediction."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temperature = sensor_data['temperature']
        humidity = sensor_data['humidity']
        risk_level = result['risk_level']
        probability = result['probability']
        
        color_codes = {
            'LOW': '\033[92m',      # Green
            'MEDIUM': '\033[93m',   # Yellow
            'HIGH': '\033[91m',     # Red
            'CRITICAL': '\033[95m'  # Magenta
        }
        reset_color = '\033[0m'
        
        color = color_codes.get(risk_level, reset_color)
        
        print(f"[{timestamp}]")
        print(f"  Temperature: {temperature:.1f}°C")
        print(f"  Humidity: {humidity:.1f}%")
        print(f"  {color}Risk Level: {risk_level} (Confidence: {probability:.2%}){reset_color}")
        print()
    
    def stop(self) -> None:
        """Stop monitoring and cleanup."""
        self.running = False
        self.data_source.disconnect()
        self._save_history()
    
    def _save_history(self) -> None:
        """Save readings history to file."""
        if not self.readings_history:
            return
        
        try:
            import json
            history_file = Path("data/sensor/readings_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            history_data = []
            for entry in self.readings_history:
                history_data.append({
                    'timestamp': entry['timestamp'].isoformat(),
                    'temperature': entry['data']['temperature'],
                    'humidity': entry['data']['humidity'],
                    'risk_level': entry['result']['risk_level'],
                    'probability': entry['result']['probability']
                })
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            print(f"✓ Readings history saved to {history_file}")
        except Exception as e:
            print(f"✗ Failed to save history: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from readings history."""
        if not self.readings_history:
            return {}
        
        temperatures = [r['data']['temperature'] for r in self.readings_history]
        humidities = [r['data']['humidity'] for r in self.readings_history]
        risk_levels = [r['result']['risk_level'] for r in self.readings_history]
        probabilities = [r['result']['probability'] for r in self.readings_history]
        
        return {
            'total_readings': len(self.readings_history),
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
            'risk_distribution': {
                level: risk_levels.count(level)
                for level in set(risk_levels)
            },
            'avg_probability': sum(probabilities) / len(probabilities)
        }


def main() -> None:
    """Main entry point for Arduino monitoring."""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    port = os.getenv('ARDUINO_PORT', '/dev/ttyUSB0')
    model_path = os.getenv('SENSOR_MODEL_PATH', 'models/saved/sensor_model.pkl')
    
    arduino_source = ArduinoDataSource(port=port)
    monitor = FireRiskMonitor(
        data_source=arduino_source,
        sensor_model_path=model_path,
        update_interval=5.0
    )
    
    monitor.start()
    
    print("\nMonitoring Statistics:")
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
