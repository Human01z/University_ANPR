// ANPR Gate Trigger (Arduino Uno)
// Sends "TRIGGER <count>" over Serial when IR beam/sensor detects a vehicle.
// Compatible with tapo_trigger_gpu.py defaults:
//   ARDUINO_BAUD = 9600
//   ARDUINO_TRIGGER_WORDS includes "TRIGGER"

// --- Pin Configuration ---
const int irSensorPin = 2;   // IR sensor OUT/DO -> D2
const int ledPin = 13;       // Built-in LED

// Most common IR modules: HIGH when idle, LOW when object detected.
const int IDLE_STATE = HIGH;
const int DETECT_STATE = LOW;

// --- Timing / Stability ---
const unsigned long cooldownDelayMs = 4000; // avoid duplicate triggers per vehicle
const unsigned long debounceMs = 50;        // suppress electrical noise

int lastSensorState = IDLE_STATE;
unsigned long lastTriggerTime = 0;
unsigned long lastStateChangeTime = 0;
int triggerCount = 0;

void setup() {
  Serial.begin(9600);
  pinMode(irSensorPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);

  // Ready blink
  digitalWrite(ledPin, HIGH);
  delay(250);
  digitalWrite(ledPin, LOW);
  Serial.println("System Ready. Waiting for beam...");
}

void loop() {
  int currentState = digitalRead(irSensorPin);
  unsigned long now = millis();

  // Track state changes for debounce timing.
  if (currentState != lastSensorState) {
    lastStateChangeTime = now;
    lastSensorState = currentState;
  }

  // Only process stable state after debounce interval.
  if ((now - lastStateChangeTime) < debounceMs) {
    delay(5);
    return;
  }

  // Fire only when sensor is in DETECT state and cooldown is over.
  if (currentState == DETECT_STATE && (now - lastTriggerTime) >= cooldownDelayMs) {
    triggerCount++;
    Serial.print("TRIGGER ");
    Serial.println(triggerCount);

    digitalWrite(ledPin, HIGH);
    delay(120);
    digitalWrite(ledPin, LOW);

    lastTriggerTime = now;
  }

  delay(10);
}

