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
# print(type(image))

# plt.imshow(image)

# # plt.subplot(image)

# plt.show()

# # # Display the image
# cv2.imshow("Image", image)
 
# # Wait for the user to press a key
# cv2.waitKey(0)
 
# # Close all windows
# cv2.destroyAllWindows()

# reading images
img_origin = image
img_B = image
img_G = image
img_R = image
  

############################################# BGR
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(img_origin)
plt.axis('on')
plt.title("BGR")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
img_B = img_B[:,:,0]
# showing image
plt.imshow(img_B, cmap=plt.cm.gray)
plt.axis('on')
plt.title("B")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
img_G = img_G[:,:,1]
# showing image
plt.imshow(img_G, cmap=plt.cm.gray)
plt.axis('on')
plt.title("G")
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
img_R = img_R[:,:,2]
# showing image
plt.imshow(img_R, cmap=plt.cm.gray)
plt.axis('on')
plt.title("R")

################################################## RGB

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 5)
img_RGB = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
# img_RGB = img_RGB[:,:,2]
# showing image
plt.imshow(img_RGB)
plt.axis('on')
plt.title("RGB")

# Adds a subplot at the 6th position
fig.add_subplot(rows, columns, 6)
img_RGB_R = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
img_RGB_R = img_RGB_R[:,:,0]
# showing image
plt.imshow(img_RGB_R, cmap=plt.cm.gray)
plt.axis('on')
plt.title("R")

# Adds a subplot at the 7th position
fig.add_subplot(rows, columns, 7)
img_RGB_G = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
img_RGB_G = img_RGB_G[:,:,1]
# showing image
plt.imshow(img_RGB_G, cmap=plt.cm.gray)
plt.axis('on')
plt.title("G")

# Adds a subplot at the 8th position
fig.add_subplot(rows, columns, 8)
img_RGB_B = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
img_RGB_B = img_RGB_B[:,:,2]
# showing image
plt.imshow(img_RGB_B, cmap=plt.cm.gray)
plt.axis('on')
plt.title("B")


# Adds a subplot at the 9th position

# r = np.array(img_origin[:,:,2], dtype = np.uint8)
# g = np.array(img_origin[:,:,1], dtype = np.uint8)
# b = np.array(img_origin[:,:,0], dtype = np.uint8)

new_image = np.array(img_origin[:,:,[2, 1, 0]], dtype = np.uint8)
fig.add_subplot(rows, columns, 9)
# # showing image
plt.imshow(new_image)
plt.axis('on')
plt.title("HAND MAKE")



plt.show()