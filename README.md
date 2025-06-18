
# Pedestrian Detection using Python and OpenCV

This project detects pedestrians (people) in an image using the HOG (Histogram of Oriented Gradients) + SVM method provided by OpenCV.

## 📸 Project Overview

We use OpenCV’s pre-trained HOG descriptor with a Linear SVM classifier to detect people in an image.

### ✅ What It Does
- Reads an image
- Uses HOG + SVM to detect people
- Applies Non-Maximum Suppression to remove overlapping boxes
- Draws bounding boxes and labels each detected person
- Displays the output image

## 🧰 Requirements

Install the following libraries:

```bash
pip install opencv-python numpy imutils
