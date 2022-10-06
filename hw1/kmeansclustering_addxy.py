import cv2  
import numpy as np
import matplotlib.pyplot as plt

#---read image----
image = cv2.imread('bird.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_xy = np.zeros((image.shape[0],image.shape[1],2))
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image_xy[i][j] = [i*255/1023, j*255/1023]      
pixel_xy = image_xy.reshape(-1, 2)
pixel_rgb = image.reshape(-1, 3)
pixel_rgb = pixel_rgb
pixel_rgbxy = np.concatenate((pixel_rgb, pixel_xy), axis = 1)
pixel_rgbxy = np.float32(pixel_rgbxy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = [2, 4, 8, 16, 32]
reconstructs = []
for _k in k:
    _, labels, centers = cv2.kmeans(pixel_rgbxy, _k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    reconstruct_rgbxy = centers[labels.flatten()]
    reconstruct_rgbxy = reconstruct_rgbxy.reshape(image.shape[0],image.shape[1],5)
    reconstruct_rgb = reconstruct_rgbxy[:,:,:3]
    reconstructs.append(reconstruct_rgb)
plt.figure(figsize=(28,23))
plt.subplot(151)
plt.imshow(reconstructs[0], cmap='gray')
plt.title("k = 2 with(x,y)")
plt.subplot(152)
plt.imshow(reconstructs[1],cmap='gray')
plt.title("k = 4 with(x,y)")
plt.subplot(153)
plt.imshow(reconstructs[2],cmap='gray')
plt.title("k = 8 with(x,y)")
plt.subplot(154)
plt.imshow(reconstructs[3],cmap='gray')
plt.title("k = 16 with(x,y)")
plt.subplot(155)
plt.imshow(reconstructs[4],cmap='gray')
plt.title("k = 32 with(x,y)")
plt.show()
plt.close()
