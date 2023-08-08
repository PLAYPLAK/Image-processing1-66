import cv2
import numpy as np
import math
import matplotlib.image as mpimg

#matplotlib notbook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image = cv2.imread("kmitl.jpg")
fig = plt.figure(figsize=(10, 10))
  
# setting values to rows and column variables
rows = 4
columns = 4

# RGB ##############################
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig.add_subplot(rows, columns, 1)  
plt.imshow(rgb_image)
plt.axis('on')
plt.title("RGB")

fig.add_subplot(rows, columns, 2)  
plt.imshow(rgb_image[:, :, 0], cmap='gray')
plt.axis('on')
plt.title("R")

fig.add_subplot(rows, columns, 3)  
plt.imshow(rgb_image[:, :, 1], cmap='gray')
plt.axis('on')
plt.title("G")

fig.add_subplot(rows, columns, 4)  
plt.imshow(rgb_image[:, :, 2], cmap='gray')
plt.axis('on')
plt.title("B")

# END RGB ##############################

# HSV ##############################
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

fig.add_subplot(rows, columns, 5)  
plt.imshow(hsv_image)
plt.axis('on')
plt.title("HSV")

fig.add_subplot(rows, columns, 6)  
plt.imshow(hsv_image[:, :, 0], cmap='gray')
plt.axis('on')
plt.title("H")

fig.add_subplot(rows, columns, 7)  
plt.imshow(hsv_image[:, :, 1], cmap='gray')
plt.axis('on')
plt.title("S")

fig.add_subplot(rows, columns, 8)  
plt.imshow(hsv_image[:, :, 2], cmap='gray')
plt.axis('on')
plt.title("V")

# END HSV ##############################

# HLS ##############################
hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

fig.add_subplot(rows, columns, 9)  
plt.imshow(hls_image)
plt.axis('on')
plt.title("HLS")

fig.add_subplot(rows, columns, 10)  
plt.imshow(hls_image[:, :, 0], cmap='gray')
plt.axis('on')
plt.title("H")

fig.add_subplot(rows, columns, 11)  
plt.imshow(hls_image[:, :, 1], cmap='gray')
plt.axis('on')
plt.title("L")

fig.add_subplot(rows, columns, 12)  
plt.imshow(hls_image[:, :, 2], cmap='gray')
plt.axis('on')
plt.title("S")

# END HLS ##############################

# YCrCb ##############################
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

fig.add_subplot(rows, columns, 13)  
plt.imshow(hls_image)
plt.axis('on')
plt.title("YCrCb")

fig.add_subplot(rows, columns, 14)  
plt.imshow(hls_image[:, :, 0], cmap='gray')
plt.axis('on')
plt.title("Y")

fig.add_subplot(rows, columns, 15)  
plt.imshow(hls_image[:, :, 1], cmap='gray')
plt.axis('on')
plt.title("Cr")

fig.add_subplot(rows, columns, 16)  
plt.imshow(hls_image[:, :, 2], cmap='gray')
plt.axis('on')
plt.title("Cb")

# YCrCb ##############################

plt.show()