import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate(image,title="Histogram"):
    hist=cv2.calcHist([image],[0],None,[256],[0,256])
    plt.plot(hist,color='black')
    plt.title(title)
    plt.xlabel("Intensity Level")
    plt.ylabel("Frequency of each Intensity")
    plt.xlim([0,256])
    plt.grid()
    plt.show()
def equalize(image):
    return cv2.equalizeHist(image)
image=cv2.imread('lion.jpg',0)
equ=equalize(image)

cv2.imshow('Original Image',image)
cv2.imshow('Equalized Image',equ)
cv2.waitKey(0)

calculate(image,"Original Image")
calculate(equ,"Equalized Histogram")
