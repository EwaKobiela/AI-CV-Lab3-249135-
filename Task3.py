import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

def negative(img):
    rows, cols = img.shape[:2]
    for row in range(0,rows):
        for col in range(0,cols):
            img[row, col] = 255 - img[row, col]
    return img

#Load image in grayscale
img = cv2.imread(r'lena.png', 0)
cv2.imshow('Original image',img)
img1 = img.copy()
img1=np.uint8(img1)
img=img*0

#Selecting some valuefor tresholding:
tr_val = 120
#Binary thresholding and showing results:
ret, img_thre = cv2.threshold(img1, tr_val, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded image', img_thre)

#Calculating and showing negative image:
img_neg = negative(img1)
cv2.imshow('Negative image', img_neg)

#Plotting image historgams
plt.hist(img_neg.ravel(),256,[0,256]); plt.title('Historgam of negative image'); plt.show()
plt.hist(img1.ravel(),256,[0,256]); plt.title('Historgam of original image'); plt.show()
plt.hist(img_thre.ravel(),256,[0,256]); plt.title('Historgam of thresholded image'); plt.show()


cv2.waitKey()
cv2.destroyAllWindows()