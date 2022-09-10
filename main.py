import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

filename = "full_body13.png"
# 예측할 그림 가져오기, gray convert
img = cv2.imread(filename)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = imutils.resize(img,
                     width=min(400, img.shape[1]))

(regions, _) = hog.detectMultiScale(img,
                                    winStride=(4, 4),
                                    padding=(4, 4),
                                    scale=1.05)

for (x, y, w, h) in regions:
    cv2.rectangle(img, (x, y),
                  (x + w, y + h),
                  (0, 0, 255), 2)

    half_of_x = x+round(w*0.5)
    half_of_y = y+round(h*0.35)

    img = cv2.line(img, (half_of_x, y), (half_of_x, y + h), (0, 0, 255), 1)
    img = cv2.line(img, (x, half_of_y), (x+w, half_of_y), (0, 0, 255), 1)

cv2.imshow('frame', img)
while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
#
# body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
#
# while(True):
#     frame = image1
#     grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     body = body_cascade.detectMultiScale(grayImage1, 1.01, 1)
#
#
#     for (x, y, w, h) in body:
#         cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 3)
#
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
