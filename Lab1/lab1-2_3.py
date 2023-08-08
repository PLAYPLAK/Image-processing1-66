import cv2
import numpy as np
import math
import matplotlib.image as mpimg

#matplotlib notbook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


image = cv2.imread("kmitl.jpg")
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# create figure
fig = plt.figure(figsize=(8, 5))
  
# setting values to rows and column variables
rows = 3
columns = 4

# create a mask
mask = np.zeros(new_image.shape[:2], np.uint8)
mask[100:250, 150:450] = 255

# compute the bitwise AND using the mask
masked_img = cv2.bitwise_and(new_image, new_image, mask = mask)

fig.add_subplot(rows, columns, 1)
plt.imshow(new_image)
plt.title("Original")

fig.add_subplot(rows, columns, 2)
plt.imshow(mask)
plt.title("Image Mask")

fig.add_subplot(rows, columns, 3)
plt.imshow(masked_img)
plt.title("Bitwise AND")

plt.show()


#link code ข้อนี้อ้างจากอันนี้นะ https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python