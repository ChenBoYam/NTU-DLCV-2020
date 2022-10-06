import cv2  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
from scipy import ndimage
#---read image----
image = imread('bird.jpg') / 255
image_xy = np.zeros((image.shape[0],image.shape[1],2))
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image_xy[i][j] = [i/(image.shape[0]-1), j/(image.shape[1]-1)]
print(image_xy.shape)       
pixel_xy = image_xy.reshape((-1, 2))
pixel_rgb = image.reshape((-1, 3))
pixel_rgbxy = np.concatenate((pixel_rgb, pixel_xy), axis = 1)
print(pixel_rgbxy.shape)   

kmeans = KMeans(n_clusters = 16,init = 'random', random_state=0).fit(pixel_rgbxy)
clustered = kmeans.cluster_centers_[kmeans.labels_]
clustered_5D = clustered.reshape(image.shape[0],image.shape[1],5)
reconstruct_rgb = clustered_5D[:,:,:3]

plt.imshow(reconstruct_rgb)
plt.show()
