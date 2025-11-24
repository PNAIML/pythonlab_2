import cv2
import time

cap = cv2.VideoCapture(0)

for x in range(1,10):
    ret, frame = cap.read()
    cv2.imshow("Frame",frame)
    ch = cv2.waitKey(800) 
    cv2.imwrite(r'C:\Users\HP\Pictures\punit_1.9.'+str(x)+'.jpg', frame)
    time.sleep(1)
    print(x)

cv2.destroyAllWindows()
cap.release()
print('Done')

