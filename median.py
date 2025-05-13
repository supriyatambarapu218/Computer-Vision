import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('img.webp')
if img is None:
    print("Image Not Found")
blur = cv2.medianBlur(img,7)
cv2.imshow("Original Image",img)
cv2.imshow("Median Filter",blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
