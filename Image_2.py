



import cv2

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read one frame from the camera
ret, frame = cap.read()

# If frame was read correctly, save it
if ret:
    cv2.imshow("Captured Image", frame)  # Optional: show the image
    cv2.imwrite(r"C:\Users\HP\Pictures\captured_image.jpg", frame)  # Save the image to file
    print("Image captured and saved as 'captured_image.jpg'")
else:
    print("Error: Could not read frame from camera.")

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
