#include <DHT.h>
#include <SoftwareSerial.h>

#define DHTPIN 2
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

const int ledGreen = 8;
const int ledYellow = 9;
const int ledRed = 10;

void setup() {
  Serial.begin(9600);
  dht.begin();
  
  pinMode(ledGreen, OUTPUT);
  pinMode(ledYellow, OUTPUT);
  pinMode(ledRed, OUTPUT);
  
  digitalWrite(ledGreen, LOW);
  digitalWrite(ledYellow, LOW);
  digitalWrite(ledRed, LOW);
}

void loop() {
  delay(2000);
  
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("ERROR");
    return;
  }
  
  Serial.print(temperature);
  Serial.print(",");
  Serial.println(humidity);
  
  delay(500);
}

void updateLEDs(float risk) {
  digitalWrite(ledGreen, LOW);
  digitalWrite(ledYellow, LOW);
  digitalWrite(ledRed, LOW);
  
  if (risk < 0.35) {
    digitalWrite(ledGreen, HIGH);
  } else if (risk < 0.65) {
    digitalWrite(ledYellow, HIGH);
  } else {
    digitalWrite(ledRed, HIGH);
  }
}