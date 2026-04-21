// --- Configuration ---
const int irSensorPin = 2; // Pin connected to the IR sensor's OUT/DO pin
const int ledPin = 13;     // Built-in LED for visual feedback on the board

// --- State Variables ---
int lastSensorState = HIGH; // Most IR modules output HIGH when idle, LOW when detecting
unsigned long lastTriggerTime = 0;

// Set a cooldown so one slow car doesn't trigger it twice
// 4000 milliseconds = 4 seconds before it can trigger again
const unsigned long cooldownDelay = 4000; 

void setup() {
  // Start serial communication at 9600 baud (MUST match Python script)
  Serial.begin(9600);
  
  pinMode(irSensorPin, INPUT);
  pinMode(ledPin, OUTPUT);
  
  // Flash the built-in LED once to show the board has rebooted and is ready
  digitalWrite(ledPin, HIGH);
  delay(500);
  digitalWrite(ledPin, LOW);
}

void loop() {
  // Read the current state of the IR sensor
  int currentState = digitalRead(irSensorPin);

  // Check if the state changed from IDLE (HIGH) to DETECTED (LOW)
  if (currentState == LOW && lastSensorState == HIGH) {
    
    // Check if the cooldown period has passed
    if (millis() - lastTriggerTime > cooldownDelay) {
      
      // --- THIS IS THE MAGIC WORD PYTHON IS WAITING FOR ---
      Serial.println("TRIGGER");
      
      // Flash the onboard LED so you know it worked visually
      digitalWrite(ledPin, HIGH); 
      delay(200);                 
      digitalWrite(ledPin, LOW);
      
      // Reset the cooldown timer
      lastTriggerTime = millis();
    }
  }

  // Save the current state for the next loop
  lastSensorState = currentState;
  
  // A tiny delay to "debounce" the sensor and stop electrical noise
  delay(50); 
}