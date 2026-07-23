/**
 * ESP8266 / ESP32 Forest Fire Telemetry Node (Direct REST API Client)
 * 
 * This sketch runs on an ESP8266 or ESP32 microcontroller. It:
 * 1. Connects to Wi-Fi.
 * 2. Reads Temperature and Humidity from a DHT11 sensor.
 * 3. Sends a JSON POST request to the Flask backend endpoint (/api/readings).
 * 4. Parses the returned threat analysis to actuate warning LEDs.
 * 
 * Required Arduino Libraries:
 * - DHT sensor library (by Adafruit)
 * - Adafruit Unified Sensor (by Adafruit)
 * - ArduinoJson (by Benoit Blanchon, supports v6 or v7)
 */

#if defined(ESP8266)
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
#elif defined(ESP32)
  #include <WiFi.h>
  #include <HTTPClient.h>
#else
  #error "This board is not supported. Please compile for ESP8266 or ESP32."
#endif

#include <WiFiClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// =========================================================================
// Configuration
// =========================================================================

// Wi-Fi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server API Endpoint
// Replace with the actual IP address of the PC running the Flask server
const char* serverUrl = "http://192.168.1.100:5000/api/readings";

// Telemetry Interval (milliseconds)
const unsigned long interval = 5000;

// DHT Sensor Settings
#define DHTPIN 4     // Pin D2 on NodeMCU/ESP8266 (GPIO 4), or Pin 4 on ESP32
#define DHTTYPE DHT11

// Warning LED Pin Configurations
#define LED_GREEN 12  // Pin D6 on ESP8266 (GPIO 12), or GPIO 12 on ESP32
#define LED_YELLOW 13 // Pin D7 on ESP8266 (GPIO 13), or GPIO 13 on ESP32
#define LED_RED 15    // Pin D8 on ESP8266 (GPIO 15), or GPIO 15 on ESP32

// =========================================================================
// Global Instances
// =========================================================================
DHT dht(DHTPIN, DHTTYPE);
unsigned long lastExecutionTime = 0;

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println("\n--- Forest Fire Detection Node Initializing ---");

  // Initialize LEDs
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  
  // Turn off all LEDs initially
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED, LOW);

  // Initialize Sensor
  dht.begin();
  Serial.println("[OK] DHT11 sensor initialized.");

  // Connect to Wi-Fi Network
  connectWiFi();
}

void loop() {
  // Check WiFi connection status and reconnect if needed
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // Non-blocking timer execution
  unsigned long currentTime = millis();
  if (currentTime - lastExecutionTime >= interval || lastExecutionTime == 0) {
    lastExecutionTime = currentTime;
    
    // 1. Read environmental data
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();

    if (isnan(temp) || isnan(hum)) {
      Serial.println("[ERR] Failed to read from DHT11 sensor.");
      triggerSensorErrorLEDs();
      return;
    }

    Serial.printf("\n[Readings] Temp: %.1f°C | Humidity: %.1f%%\n", temp, hum);

    // 2. Send POST request with JSON payload
    sendTelemetry(temp, hum);
  }
}

// =========================================================================
// Helper Functions
// =========================================================================

void connectWiFi() {
  Serial.print("[WiFi] Connecting to: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
    
    // Toggle Yellow LED while connecting
    digitalWrite(LED_YELLOW, !digitalRead(LED_YELLOW));
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(LED_YELLOW, LOW); // Turn off indicator
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WiFi] Connection failed. Re-trying next cycle.");
    triggerSensorErrorLEDs();
  }
}

void sendTelemetry(float temp, float hum) {
  WiFiClient client;
  HTTPClient http;

  Serial.println("[HTTP] Connecting to API server...");
  
  if (http.begin(client, serverUrl)) {
    http.addHeader("Content-Type", "application/json");

    // Create JSON Document (requires about 96 bytes of stack allocation)
    StaticJsonDocument<128> doc;
    doc["temperature"] = temp;
    doc["humidity"] = hum;

    String requestBody;
    serializeJson(doc, requestBody);

    // Send HTTP POST Request
    int httpResponseCode = http.POST(requestBody);
    
    if (httpResponseCode > 0) {
      Serial.printf("[HTTP] POST Response Code: %d\n", httpResponseCode);
      
      if (httpResponseCode == HTTP_CODE_OK || httpResponseCode == 201) {
        String responseBody = http.getString();
        parseAPIResponse(responseBody);
      } else {
        Serial.println("[HTTP] Server returned error status.");
        triggerSensorErrorLEDs();
      }
    } else {
      Serial.printf("[HTTP] POST failed. Error: %s\n", http.errorToString(httpResponseCode).c_str());
      triggerSensorErrorLEDs();
    }
    
    http.end();
  } else {
    Serial.println("[HTTP] Unable to establish connection to server URL.");
    triggerSensorErrorLEDs();
  }
}

void parseAPIResponse(String jsonString) {
  // Allocate memory for parsing response JSON
  StaticJsonDocument<256> responseDoc;
  DeserializationError error = deserializeJson(responseDoc, jsonString);

  if (error) {
    Serial.print("[JSON] Deserialization failed: ");
    Serial.println(error.f_str());
    triggerSensorErrorLEDs();
    return;
  }

  // Validate backend response fields
  bool success = responseDoc["success"] | false;
  if (!success) {
    Serial.println("[API] Warning: Server response indicated unsuccessful prediction.");
    return;
  }

  float riskScore = responseDoc["confidence"] | 0.0;
  const char* riskLevel = responseDoc["risk_level"] | "UNKNOWN";
  const char* recommendation = responseDoc["recommendation"] | "N/A";

  Serial.printf("[API] Risk Probability: %.1f%%\n", riskScore * 100.0);
  Serial.printf("[API] Risk Level: %s\n", riskLevel);
  Serial.printf("[API] Recommendation: %s\n", recommendation);

  // Update LED states based on risk probability
  updateLEDs(riskScore);
}

void updateLEDs(float risk) {
  // Turn off all warning LEDs
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED, LOW);

  if (risk < 0.35) {
    // Low risk: Green LED ON
    digitalWrite(LED_GREEN, HIGH);
  } else if (risk < 0.65) {
    // Medium risk: Yellow LED ON
    digitalWrite(LED_YELLOW, HIGH);
  } else {
    // High/Critical risk: Red LED ON
    digitalWrite(LED_RED, HIGH);
  }
}

void triggerSensorErrorLEDs() {
  // Error state: Flashing all LEDs simultaneously
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LED_YELLOW, HIGH);
  digitalWrite(LED_RED, HIGH);
  delay(100);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED, LOW);
}
