import cv2

img=cv2.imread(r"C:\Users\HP\Pictures\Saved Pictures\TEMPLE_PUNE.jpg")
if img is None:
    print("Failed to load image")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces=face_cascade.detectMultiScale(gray,1.1,2)
for (x,y,w,h)in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
