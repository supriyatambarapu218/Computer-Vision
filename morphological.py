import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
image = cv2.imread('lion.jpg', 0)  # Change 'input_image.jpg' to your image file path
image=cv2.resize(image,(400,400))

# Define a kernel (structuring element)
kernel = np.ones((5,5), np.uint8)

# Erosion
erosion = cv2.erode(image, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(image, kernel, iterations=1)

# Opening (erosion followed by dilation)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Original",image)
cv2.imshow("Erosion",erosion)
cv2.imshow("Dilation",dilation)
cv2.imshow("Opening",opening)
cv2.imshow("Closing",closing) 

cv2.waitKey(0)
cv2.destroyAllWindows()
