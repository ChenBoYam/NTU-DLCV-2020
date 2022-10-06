from PIL import Image
import numpy
import scipy
import math
import time
import matplotlib.pyplot as plt

i = Image.open('lena.png')
width, height = i.size

im = numpy.array(Image.open('lena.png'))
im1 = Image.fromarray(im, 'L')
im1.show()

convxr=numpy.array([1, 0, -1])

convxc=numpy.array([[1],
                    [2],
                    [1]])

convyr=numpy.array([1, 2, 1])

convyc=numpy.array([[1],
                    [0],
                    [-1]])

image = numpy.zeros(shape=(width + 2, height + 2), dtype=int)
image1 = numpy.array(image)

for m in range(0, width - 1):
    for n in range(0, height - 1):
        image1[m + 1, n + 1] = im[m, n]

def conv1d(w, h, image, convc, convr):
    image2 = numpy.zeros(shape=(w, h), dtype=int)
    image2 = numpy.array(image2)
    for x in range(1, w - 2):
        for y in range(1, h - 2):
            image2[x, y] = (image[x - 1, y] * convc[0, 0] + image[x, y] * convc[1,0] + image[x + 1, y] * convc[2, 0])
    image3 = numpy.zeros(shape=(width, height), dtype=int)
    image3 = numpy.array(image3)
    for x in range(1, w - 2):
        for y in range(1, h - 2):
            image3[x-1, y-1] = (image2[x, y-1] * convr[0] + image2[x, y] * convr[1] + image2[x, y+1] * convr[2])
            if (image3[x, y] < 0):
                image3[x, y] =-image3[x,y]
    return image3

G1d = numpy.zeros(shape=(width, height), dtype=int)
G1d = numpy.array(G1d)

G1dx=conv1d(width + 2, height + 2, image1, convxc, convxr)
G1dy=conv1d(width + 2, height + 2, image1, convyc, convyr)

plt.figure(figsize=(64,64))
plt.subplot(121)
plt.imshow(G1dx, cmap='gray', vmin=0, vmax=255)
plt.title("I x")
plt.subplot(122)
plt.imshow(G1dy, cmap='gray', vmin=0, vmax=255)
plt.title("I y")
plt.show()
plt.close()


