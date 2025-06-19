# 🚶‍♂️ Pedestrian Detection using OpenCV

This project uses OpenCV's HOG (Histogram of Oriented Gradients) + SVM (Support Vector Machine) method to detect people in an image.

## 📸 How It Works

- Loads and resizes an image.
- Detects people using HOGDescriptor with a pre-trained SVM detector.
- Uses Non-Maxima Suppression to remove overlapping boxes.
- Draws rectangles and labels each detected person.
- Displays the final image with all detected people.

## 🛠 Requirements

Install the required libraries using pip:

```bash
pip install opencv-python numpy imutils
