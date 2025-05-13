import cv2
import numpy as np
from matplotlib import pyplot as plt
 
image = cv2.imread('flower.webp')
img=cv2.resize(image,(400,400))
if img is None:
    print("Image Not Found")
blur = cv2.GaussianBlur(img,(5,5),12)
cv2.imshow("Original Image",img)
cv2.imshow("Gaussian Filter",blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
