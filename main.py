import cv2
import imutils

filename = "full_body13.png"
img = cv2.imread(filename)
# get img

hog = cv2.HOGDescriptor()
# get HOG algorithm
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# input data in svm with HOG algorithm which detecting people with cv

img = imutils.resize(img,
                     width=min(400, img.shape[1]))
# resize img 400 or img.size whichever is smaller

(regions, _) = hog.detectMultiScale(img,
                                    winStride=(4, 4),
                                    padding=(4, 4),
                                    scale=1.05)
#  detect mutli people from img. with winStride(4,4),
#  winStride : 2-tuple that dictats the step size
#  padding : surround the object

for (x, y, w, h) in regions:
    # for each human

    cv2.rectangle(img, (x, y),
                  (x + w, y + h),
                  (0, 0, 255), 2)
    # draw rectangle

    half_of_x = x + round(w * 0.5)
    half_of_y = y + round(h * 0.35)

    img = cv2.line(img, (half_of_x, y), (half_of_x, y + h), (0, 0, 255), 1)
    img = cv2.line(img, (x, half_of_y), (x + w, half_of_y), (0, 0, 255), 1)
    # draw line

cv2.imshow('frame', img)
#  show frame

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # wait key 'q'

cv2.destroyAllWindows()
# destroy all windows
