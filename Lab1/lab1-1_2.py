import cv2
import numpy as np
import math
import matplotlib.image as mpimg


#matplotlib notbook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create figure
fig = plt.figure(figsize=(10, 5))
  
# setting values to rows and column variables
rows = 3
columns = 4

############################################

image = cv2.imread("lab1/kmitl.jpg")
new_image = np.array(image[:,:,[2, 1, 0]], dtype = np.uint8)
print("Default= ")
print(image.shape)

#Nomal ################################# 
fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(new_image[:,:,0])
plt.axis('on')
plt.title(f"Default")

#Tranpose ################################# 
fig.add_subplot(rows, columns, 2)
# showing image
image_tranpose = np.transpose(new_image)
print("Tranposed = ")
print(image_tranpose.shape)
plt.imshow(image_tranpose[0,: , :])
plt.axis('on')
plt.title("Tranpose")

#MoveAxis ################################# 
fig.add_subplot(rows, columns, 3)

image_moveaxis = np.moveaxis(new_image, -1, 0)
# showing image
print("MoveAxist = ")
print(image_moveaxis.shape)
plt.imshow(image_moveaxis[0, :, :])
plt.axis('on')
plt.title("Move-Axist")

# ReShape ################################# 
fig.add_subplot(rows, columns, 4)

image_reshape = np.reshape(new_image, (new_image.shape[2], new_image.shape[0], -1))
# showing image
print("ReShape = ")
print(image_reshape.shape)
plt.imshow(image_reshape[0,:,:])
plt.axis('on')
plt.title("Re-Shape")

plt.show()