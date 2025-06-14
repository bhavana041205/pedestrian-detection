import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

# Load the HOG descriptor/person detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Read the image and resize if too big
frame = cv2.imread("image.jpg")
width = frame.shape[1]
max_width = 600
if width > max_width:
    frame = imutils.resize(frame, width=max_width)

# Detect people in the image
(rects, weights) = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
rects = non_max_suppression(rects, probs=None, overlapThresh=0.5)

# Draw rectangles around detected people
count = 0
for (xA, yA, xB, yB) in rects:
    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
    cv2.putText(frame, f'P{count}', (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    count += 1

# Show the output image
cv2.imshow("output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
