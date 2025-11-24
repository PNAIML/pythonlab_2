import cv2
import os

# Load the video
video_path = r'C:\Users\HP\Downloads\video_1.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Create a directory to save the frames


output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

frame_number = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    # Save every frame as an image
    frame_filename = os.path.join(output_dir, f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_number += 1

cap.release()
print(f"Extracted {frame_number} frames to '{output_dir}'")
