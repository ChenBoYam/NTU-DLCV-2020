import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

imgName = "lena.png"
width = 512
height = 512
kernel_size = (3, 3)
sigma = 0.7213

img1 = cv2.imread(imgName)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img1, kernel_size, sigma)

img1_sobelx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize = 3)
img1_sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize = 3)
img2_sobelx = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize = 3)
img2_sobely = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize = 3)


img3 = np.zeros(shape=(width, height), dtype=int)
img3 = np.array(img3)
img4 = np.zeros(shape=(width, height), dtype=int)
img4 = np.array(img4)

for x in range(0, 512):
        for y in range(0, height-1):
            img3[x,y] = math.sqrt(img1_sobelx[x,y]*img1_sobelx[x,y]+img1_sobely[x,y]*img1_sobely[x,y])
for x in range(0, 512):
        for y in range(0, height-1):
            img4[x,y] = math.sqrt(img2_sobelx[x,y]*img2_sobelx[x,y]+img2_sobely[x,y]*img2_sobely[x,y])

plt.figure(figsize=(64,64))
plt.subplot(121)
plt.imshow(img1, cmap='gray')
plt.title("origin")
plt.subplot(122)
plt.imshow(img2, cmap='gray')
plt.title("blurred")
plt.show()
plt.close()

plt.figure(figsize=(64,64))
plt.subplot(121)
plt.imshow(img3, cmap='gray')
plt.title("origin gradient magnitude")
plt.subplot(122)
plt.imshow(img4, cmap='gray')
plt.title("blurred gradient magnitude")
plt.show()
