import cv2
from random import randrange
import time

camera = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("frontal.xml")

frame_height = int(camera.get(4))
frame_width = int(camera.get(3))


videoWrite = cv2.VideoWriter('capture.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
while True:
    success, img = camera.read()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = cascade.detectMultiScale(imgGray)

    for (x, y, w, h) in face_coordinates:
        top = (x, y)
        bot = (x + w, y + h)
        font = cv2.FONT_HERSHEY_DUPLEX 
        img = cv2.rectangle(img, top, bot, (randrange(256), randrange(256), randrange(256)), 2)
    if success == True: videoWrite.write(img)
    cv2.imshow('facedetect', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
videoWrite.release()
cv2.destroyAllWindows()