import face_recognition
import cv2
import os

# Step 1: Load and encode all faces from the dataset
dataset_dir = r"C:\Users\HP\Pictures\dataset"
known_encodings = []
known_names = []

for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(dataset_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)

# Step 2: Load the prediction image and extract its encoding
unknown_img = face_recognition.load_image_file(r"C:\Users\HP\Pictures\PunitTest.jpg")
unknown_encodings = face_recognition.face_encodings(unknown_img)

# Step 3: Compare faces
for unknown_encoding in unknown_encodings:
    results = face_recognition.compare_faces(known_encodings, unknown_encoding)
    if True in results:
        matched_idx = results.index(True)
        print(f"Predicted person: {known_names[matched_idx]}")
    else:
        print("No match found.")

