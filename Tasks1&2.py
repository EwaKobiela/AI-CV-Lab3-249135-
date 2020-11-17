import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

def quantize(img, lvl):
    image = np.float32(img)
    step = 256//lvl
    list = np.zeros(lvl, dtype=int)
    for s in range(0,lvl):
        list[s]=s*step
    rows, cols = img.shape[:2]
    for row in range(0, cols):
        for col in range(0, cols):
            val = img[row, col]
            new_val = list.flat[np.abs(list - val).argmin()]
            img[row,col]=new_val
    return img
def equalize(img):
    equ=cv2.equalizeHist(img)
    return equ
def stretch(img):
    rows, cols = img.shape[:2]
    max_val = np.max(img)
    min_val = np.min(img)
    for row in range(0, cols):
        for col in range(0, cols):
            val = img[row, col]
            new_val = ((val-min_val)/(max_val-min_val))*255
            img[row, col] = new_val
    return img

#Load images
img = cv2.imread(r'lena.png',0)
#cv2.imshow('High frequency image',img)
imgHighFreq = img.copy()
img=img*0
img2 = cv2.imread(r'lowfreq.png',0)
cv2.imshow('Low frequency image',img2)
imgLowFreq = img2.copy()
img2=img2*0

#Define qualtization level and perform quantization
lvl_of_intensity = 32     #32, 64 or 128
#quant = quantize(imgHighFreq, lvl_of_intensity)
#OR
quant = quantize(imgLowFreq, lvl_of_intensity)
cv2.imshow('Quantized image',quant)

#Stretch histogram for imgs (Task 1)
str = stretch(quant)
cv2.imshow("Stretched histogram", str)

#Equalize historgam for imgs (Task 2)
#for base images:
#high_eq = equalize(imgHighFreq)
#cv2.imshow("Original image with equalized histogram", high_eq)
#OR
low_eq = equalize(imgLowFreq)
cv2.imshow("Original image with equalized histogram", low_eq)
#for quantized image:
equ = equalize(quant)
cv2.imshow("Quantized image with equalized histogram", equ)

cv2.waitKey()
cv2.destroyAllWindows()