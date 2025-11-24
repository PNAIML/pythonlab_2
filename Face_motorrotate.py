import cv2
import RPi.GPIO as GPIO

motor_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_pin, GPIO.OUT)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        GPIO.output(motor_pin, GPIO.HIGH)  # Rotate motor
    else:
        GPIO.output(motor_pin, GPIO.LOW)   # Stop motor
        # Optional: Show the frame with detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Face Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
# Cleanup
    cap.release()
    cv2.destroyAllWindows()
    GPIO.output(motor_pin, GPIO.LOW)
    GPIO.cleanup()
