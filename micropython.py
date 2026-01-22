from machine import Pin, PWM, UART
import time

# ====== SERVO SETUP ======
servo_pin = PWM(Pin(15))
servo_pin.freq(50)  # standard servo frequency

def set_servo_angle(angle):
    # angle: 0–180
    min_us = 500
    max_us = 2500
    us = min_us + (max_us - min_us) * angle / 180
    duty = int(us * 1023 / 20000)  # for 20ms period
    servo_pin.duty(duty)

# ====== DC MOTOR SETUP (H-bridge) ======
motor_in1 = Pin(12, Pin.OUT)
motor_in2 = Pin(13, Pin.OUT)

def motor_forward():
    motor_in1.value(1)
    motor_in2.value(0)

def motor_stop():
    motor_in1.value(0)
    motor_in2.value(0)

# ====== UART SETUP ======
uart = UART(0, baudrate=9600)  # adjust to your board

# ====== STATE ======
blocked_mode = False

def sweep_servo():
    # simple left-right sweep
    for angle in range(40, 140, 5):
        set_servo_angle(angle)
        time.sleep(0.03)
        if uart.any():
            return
    for angle in range(140, 40, -5):
        set_servo_angle(angle)
        time.sleep(0.03)
        if uart.any():
            return

# ====== MAIN LOOP ======
set_servo_angle(90)  # center
motor_forward()       # start moving

while True:
    if uart.any():
        cmd = uart.read(1)
        if not cmd:
            continue
        c = cmd.decode('utf-8')

        if c == 'B':
            # Blocked path: stop and start scanning
            blocked_mode = True
            motor_stop()
            print("Received B: BLOCKED, starting servo sweep")

        elif c == 'G':
            # Unblocked yellow path: go straight
            blocked_mode = False
            set_servo_angle(90)
            motor_forward()
            print("Received G: GO, centered servo and moving forward")

        elif c == 'N':
            # No yellow: you can choose to keep sweeping if blocked_mode
            print("Received N: NO YELLOW (continue current behavior)")

    if blocked_mode:
        sweep_servo()
    else:
        time.sleep(0.01)
