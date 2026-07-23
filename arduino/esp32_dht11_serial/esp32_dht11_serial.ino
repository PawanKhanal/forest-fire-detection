/**
 * ESP32 Forest Fire Telemetry Node (Serial USB Version)
 * 
 * This sketch runs on an ESP32 microcontroller connected via USB Serial. It:
 * 1. Reads Temperature and Humidity from a DHT11 sensor.
 * 2. Writes environmental data to Serial as "T:temp,H:humidity\n".
 * 3. Reads computed threat analysis probability back from Serial.
 * 4. Actuates the Blue LED (Safe/Low Risk) or Red LED (High/Critical Risk).
 * 
 * Required Arduino Libraries:
 * - DHT sensor library (by Adafruit)
 * - Adafruit Unified Sensor (by Adafruit)
 */

#include <DHT.h>

// =========================================================================
// Configuration
// =========================================================================

// Telemetry Interval (milliseconds)
const unsigned long interval = 5000;

// DHT Sensor Settings
#define DHTPIN 4      // DHT11 Data Pin connected to GPIO 4 on ESP32
#define DHTTYPE DHT11

// Warning LED Pin Configurations (Using Blue and Red LEDs only)
#define LED_BLUE 12   // GPIO 12 on ESP32 for Safe / Low Risk
#define LED_RED 15    // GPIO 15 on ESP32 for High / Critical Risk

// Timeout for receiving serial communication from PC (in milliseconds)
const unsigned long serialTimeout = 15000; 

// =========================================================================
// Global Instances
// =========================================================================
DHT dht(DHTPIN, DHTTYPE);
unsigned long lastExecutionTime = 0;
unsigned long lastSerialReceiveTime = 0;
bool pcConnected = false;

void setup() {
  // Initialize USB Serial communication (Baud rate matches .env ARDUINO_BAUDRATE)
  Serial.begin(9600);
  delay(10);
  
  // Initialize LEDs
  pinMode(LED_BLUE, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  
  // Flash LEDs on startup to verify hardware
  digitalWrite(LED_BLUE, HIGH);
  digitalWrite(LED_RED, HIGH);
  delay(1000);
  digitalWrite(LED_BLUE, LOW);
  digitalWrite(LED_RED, LOW);

  // Initialize DHT11 Sensor
  dht.begin();
  
  lastSerialReceiveTime = millis();
}

void loop() {
  unsigned long currentTime = millis();

  // 1. Periodically read DHT11 and send data to PC
  if (currentTime - lastExecutionTime >= interval || lastExecutionTime == 0) {
    lastExecutionTime = currentTime;
    
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();

    if (isnan(temp) || isnan(hum)) {
      // Blink both LEDs if DHT11 sensor fails to read
      triggerSensorErrorLEDs();
    } else {
      // Send formatting expected by arduino_reader.py
      Serial.print("T:");
      Serial.print(temp, 1);
      Serial.print(",H:");
      Serial.println(hum, 1);
    }
  }

  // 2. Read risk probability feedback from the Python app via Serial
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');
    inputString.trim();
    
    if (inputString.length() > 0) {
      float riskProbability = inputString.toFloat();
      pcConnected = true;
      lastSerialReceiveTime = millis();
      
      // Update LED warning states based on risk probability
      updateLEDs(riskProbability);
    }
  }

  // 3. Fallback: If no serial communication received from PC for too long
  if (pcConnected && (millis() - lastSerialReceiveTime > serialTimeout)) {
    pcConnected = false;
    // Turn off normal indicators and slow blink to show connection loss
    digitalWrite(LED_BLUE, LOW);
    digitalWrite(LED_RED, LOW);
  }
}

// Update LEDs based on risk
void updateLEDs(float risk) {
  digitalWrite(LED_BLUE, LOW);
  digitalWrite(LED_RED, LOW);

  // Threshold: 0.50 (corresponds to HIGH or CRITICAL risk)
  if (risk < 0.50) {
    // Low / Medium Risk: Blue LED ON
    digitalWrite(LED_BLUE, HIGH);
  } else {
    // High / Critical Risk: Red LED ON
    digitalWrite(LED_RED, HIGH);
  }
}

// Fast toggle of both LEDs to signal sensor read failure
void triggerSensorErrorLEDs() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BLUE, HIGH);
    digitalWrite(LED_RED, HIGH);
    delay(150);
    digitalWrite(LED_BLUE, LOW);
    digitalWrite(LED_RED, LOW);
    delay(150);
  }
}
