import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('flower.webp') 
inverse = 255 - image
plt.imshow(inverse)
