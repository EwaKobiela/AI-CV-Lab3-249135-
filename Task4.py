import cv2
import numpy as np
from PIL import Image as im
import imutils
from matplotlib import pyplot as plt

#Loading image in grayscale
img = cv2.imread(r'lena.png', 0)
img1 = img.copy()
img1=np.uint8(img1)
img=img*0
#Showing original image
plt.subplot(221),plt.imshow(img1, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

#Computing DFT
dft = cv2.dft(np.float32(img1),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#Showing DFT results
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('DFT results'), plt.xticks([]), plt.yticks([])

#Calculating image dimensions
row = img1.shape[0]
col = img1.shape[1]
mid_row = row // 2
mid_col = col // 2
#Creating mask (centre pixel = 0, rest = 1)
mask = np.ones((row, col, 2), np.uint8)
mask[mid_row:mid_row, mid_col:mid_col] = 0
#Applying mask to DFT image
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
#Inverse DFT transformation
res = cv2.idft(f_ishift)
res = cv2.magnitude(res[:, :, 0], res[:, :, 1])
#Showing inverse transformation results
plt.subplot(223),plt.imshow(res, cmap = 'gray')
plt.title('Restored image'), plt.xticks([]), plt.yticks([])

plt.show()