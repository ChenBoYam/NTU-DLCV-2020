import cv2  
import numpy as np
import matplotlib.pyplot as plt

#---read image----
image = cv2.imread('bird.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_rgb = image.reshape((-1, 3))
pixel_rgb = np.float32(pixel_rgb)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = [2, 4, 8, 16, 32]
reconstructs = []
for _k in k:
    _, labels, centers = cv2.kmeans(pixel_rgb, _k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    reconstruct_rgb = centers[labels.flatten()]
    reconstruct = reconstruct_rgb.reshape(image.shape)
    reconstructs.append(reconstruct)


plt.figure(figsize=(28,23))
plt.subplot(151)
plt.imshow(reconstructs[0], cmap='gray')
plt.title("k = 2, k++")
plt.subplot(152)
plt.imshow(reconstructs[1],cmap='gray')
plt.title("k = 4, k++")
plt.subplot(153)
plt.imshow(reconstructs[2],cmap='gray')
plt.title("k = 8, k++")
plt.subplot(154)
plt.imshow(reconstructs[3],cmap='gray')
plt.title("k = 16, k++")
plt.subplot(155)
plt.imshow(reconstructs[4],cmap='gray')
plt.title("k = 32, k++")
plt.show()
plt.close()
