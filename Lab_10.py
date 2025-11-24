import cv2
import numpy as np
import os

# Path to your dataset
dataset_path = r'C:\Users\HP\Pictures\dataset'  # Modify if needed
model_output_path = r'C:\Users\HP\Pictures\STORE ELEX\trainer.xml'

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Storage for training data
roiList = []
idList = []

# Process each image file
for filename in os.listdir(dataset_path):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    try:
        # Expecting filename like: name.id.jpg
        id = int(filename.split('.')[1])
    except (IndexError, ValueError):
        print(f"Skipping file {filename}: cannot extract ID.")
        continue

    img_path = os.path.join(dataset_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Skipping {filename}: couldn't read image.")
        continue

    faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        roiList.append(roi)
        idList.append(id)

# Train and save the model
if roiList and idList:
    recognizer.train(roiList, np.array(idList))
    recognizer.save(model_output_path)
    print(f"[INFO] Training complete. Model saved to {model_output_path}.")
    print(f"[INFO] Trained on {len(np.unique(idList))} unique IDs.")
else:
    print("[WARNING] No valid faces found. Training aborted.")
