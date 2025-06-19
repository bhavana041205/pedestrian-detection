import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

# Load the HOG detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load and resize the image
frame = cv2.imread("image.jpg")
frame = imutils.resize(frame, width=min(800, frame.shape[1]))

# Detect people
(rects, weights) = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
rects = non_max_suppression(rects, probs=None, overlapThresh=0.5)

# Draw boxes
for i, (xA, yA, xB, yB) in enumerate(rects):
    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
    cv2.putText(frame, f'Person {i+1}', (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Show image
cv2.imshow("output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

