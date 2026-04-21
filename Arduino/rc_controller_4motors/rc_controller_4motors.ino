#include <SoftwareSerial.h>
#include <AFMotor_R4.h>

AF_DCMotor FL(1);  // M1
AF_DCMotor RL(2);  // M2
AF_DCMotor FR(3);  // M3
AF_DCMotor RR(4);  // M4
SoftwareSerial BT( A0, A1 ); // RX = A4, TX = A5

char command;

void setup() {
  Serial.begin(9600);
  BT.begin(9600);
  Serial.println("4 Motor RC Car Test");

  setAllSpeed(160);
  stopMotors();
}

void loop() {
  if (BT.available()) {
    command = BT.read();
    Serial.print("Se recibio: ");
    Serial.println(command);

    switch (command) {
      case 'F':   // Forward
      case 'f':
        forward();
        break;

      case 'B':   // Backward
      case 'b':
        backward();
        break;

      case 'L':   // Turn Left
      case 'l':
        turnLeft();
        break;

      case 'R':   // Turn Right
      case 'r':
        turnRight();
        break;

      case 'S':   // Stop
      case 's':
        stopMotors();
        break;

      case 'z':   // turbo
      case 'Z':
        setAllSpeed(255);
        break;

      case 'w':   // normal speed
      case 'W':
        setAllSpeed(160);
        break;

      default:
        Serial.println("error de comando");
        break;

       stopMotors();
    }
  }
}

void setAllSpeed(int speed) {
  FL.setSpeed(speed);
  RL.setSpeed(speed);
  FR.setSpeed(speed);
  RR.setSpeed(speed);
}

void forward() {
  FL.run(FORWARD);
  RL.run(FORWARD);
  FR.run(FORWARD);
  RR.run(FORWARD);
  Serial.println("→ Adelante");
  
}

void backward() {
  FL.run(BACKWARD);
  RL.run(BACKWARD);
  FR.run(BACKWARD);
  RR.run(BACKWARD);
  Serial.println("← Atras");
}

void turnLeft() {
  // Left wheels backward, Right wheels forward = sharp left turn
  FR.run(BACKWARD);
  RR.run(BACKWARD);
  //motor3.run(BACKWARD);
  //motor4.run(BACKWARD);
  Serial.println("↺ Izquierda");
}

void turnRight() {
  // Left wheels forward, Right wheels backward = sharp right turn
  FL.run(BACKWARD);
  RL.run(BACKWARD);
  //motor3.run(FORWARD);
  //motor4.run(FORWARD);
  Serial.println("↻ Derecha");
}

void stopMotors() {
  FL.run(RELEASE);
  RL.run(RELEASE);
  FR.run(RELEASE);
  RR.run(RELEASE);
  Serial.println("■ All motors stopped");
}
