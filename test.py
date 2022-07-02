import cv2
from detect import detect,textToSpeech
from time import sleep
import os
cam = cv2.VideoCapture(0)
cam.set(3,1080)
cam.set(4,720)
cam.set(10,150)

cv2.namedWindow("test")

img_counter = 0
textToSpeech("VISION TURNING ON")
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        cv2.imwrite("detect_1.png",frame)
        img_name = "detect_1.png"
        detect(img_name)

cam.release()

cv2.destroyAllWindows()