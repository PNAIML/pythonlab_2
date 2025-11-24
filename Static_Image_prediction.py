import face_recognition
import dlib
import cmake

# Load known image (training)
known_image = face_recognition.load_image_file(r"C:\Users\HP\Pictures\captured_image.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load unknown image (prediction)
unknown_image = face_recognition.load_image_file(r"C:\Users\HP\Pictures\PunitTest.jpg")
unknown_encodings = face_recognition.face_encodings(unknown_image)

# Compare faces
for unknown_encoding in unknown_encodings:
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    if results[0]:
        print("🎉 Face matched!")
    else:
        print("❌ Face not matched.")
