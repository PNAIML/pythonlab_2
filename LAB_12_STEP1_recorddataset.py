import cv2
import os

# Initialize camera
cap = cv2.VideoCapture(0)

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(r'C:\Users\HP\Pictures\haarcascade_frontalface_default.xml')

# Person ID and save path
person_id = input("Enter Person ID (a number): ")
save_path = r'C:\Users\HP\Pictures\dataset1'
os.makedirs(save_path, exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        filename = f"{save_path}/User.{person_id}.{count}.jpg"
        cv2.imwrite(filename, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"Capturing {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Capture Faces', frame)

    if count >= 100 or cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
print("Dataset collection completed.")
