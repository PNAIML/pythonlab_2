import cv2

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(r'C:\Users\HP\Pictures\haarcascade_frontalface_default.xml')

# Load trained face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'C:\Users\HP\Pictures\recognizer.xml')

# Start video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Font for displaying name
font = cv2.FONT_HERSHEY_SIMPLEX

# Loop through each frame of the video stream
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and predict face
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)

        # Confidence threshold can be adjusted
        if confidence < 80:
            text = f"Predicted Person ID: {id_} ({round(confidence, 2)}%)"
        else:
            text = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), font, 0.9, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Face Prediction', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
