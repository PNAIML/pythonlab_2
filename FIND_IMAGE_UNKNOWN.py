import cv2
import os

def load_images(folder):
    images = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, 0)  # grayscale
        if img is not None:
            images.append(img)
    return images

def extract_orb_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def compare_descriptors(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

# Load known images
known_images = load_images(r'C:\Users\HP\Documents\PUNIT')
known_descriptors = [extract_orb_features(img) for img in known_images]

# Load test image
test_image = cv2.imread(r"C:\Users\HP\Pictures\1231.jpg", 0)
test_des = extract_orb_features(test_image)

# Compare with known images
threshold = 30  # adjust based on experimentation
recognized = False

for des in known_descriptors:
    if des is not None and test_des is not None:
        match_count = compare_descriptors(des, test_des)
        print(f"Matches: {match_count}")
        if match_count > threshold:
            recognized = True
            break

if recognized:
    print("✅ Similar object recognized!")
else:
    print("❌ No similar object found.")
