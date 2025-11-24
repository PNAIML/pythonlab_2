import cv2
import numpy as np
from PIL import Image
import os

# Path to dataset
path = r'C:\Users\HP\Pictures\dataset'

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Haar face detector
face_cascade = cv2.CascadeClassifier(r'C:\Users\HP\Pictures\haarcascade_frontalface_default.xml')

# Get image paths
image_paths = [os.path.join(path, f) for f in os.listdir(path)]

faces = []
ids = []

for image_path in image_paths:
    img = Image.open(image_path).convert('L')  # grayscale
    img_np = np.array(img, 'uint8')
    id = int(os.path.split(image_path)[-1].split('.')[1])

    face_regions = face_cascade.detectMultiScale(img_np)

    for (x, y, w, h) in face_regions:
        faces.append(img_np[y:y+h, x:x+w])
        ids.append(id)

# Train recognizer and save model
recognizer.train(faces, np.array(ids))
recognizer.save(r'C:\Users\HP\Pictures\recognizer.xml')

print("Training complete. Model saved as recognizer.xml.")
