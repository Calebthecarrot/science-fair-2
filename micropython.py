from gpiozero import Servo, Motor
from serial import Serial
import time

# ============================
# SERVO SETUP (GPIO pins)
# ============================
# Adjust min/max pulse width for your servos
top_servo = Servo(17, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
bottom_servo = Servo(27, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

def angle_to_servo_value(angle):
    # Convert 0–180° → -1 to +1
    return (angle - 90) / 90

current_top_angle = 90
top_servo.value = angle_to_servo_value(current_top_angle)
bottom_servo.value = 0  # centered

# ============================
# DC MOTOR SETUP
# ============================
motor = Motor(forward=22, backward=23)

def motor_forward():
    motor.forward()

def motor_stop():
    motor.stop()

# ============================
# SERIAL SETUP
# ============================
ser = Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

# ============================
# CAMERA SWEEP FUNCTION
# ============================
def sweep_top_servo():
    global current_top_angle
    for angle in range(40, 140, 4):
        current_top_angle = angle
        top_servo.value = angle_to_servo_value(angle)
        time.sleep(0.03)
        if ser.in_waiting:
            return
    for angle in range(140, 40, -4):
        current_top_angle = angle
        top_servo.value = angle_to_servo_value(angle)
        time.sleep(0.03)
        if ser.in_waiting:
            return

# ============================
# MAIN LOOP
# ============================
blocked_mode = False
motor_forward()

while True:
    if ser.in_waiting:
        c = ser.read(1).decode('utf-8')

        # BLOCKED OR NO YELLOW
        if c == 'B' or c == 'N':
            blocked_mode = True
            motor_stop()
            print("STOP: Blocked or No Yellow")

        # GOOD TO GO
        elif c == 'G':
            blocked_mode = False
            bottom_servo.value = angle_to_servo_value(current_top_angle)
            motor_forward()
            print("GO: Aligning bottom servo to", current_top_angle)

    # If blocked, keep scanning
    if blocked_mode:
        sweep_top_servo()
    else:
        time.sleep(0.01)
