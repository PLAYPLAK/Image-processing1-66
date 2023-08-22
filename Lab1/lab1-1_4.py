import cv2
import numpy as np
import math
import matplotlib.image as mpimg

#matplotlib notbook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image in grayscale
image = cv2.imread('lab1/kmitl.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the image to 200x200 pixels
width, height = 200, 200
resized_image = cv2.resize(image, (width, height))

# Create meshgrid for X, Y coordinates
x_coords = np.arange(0, resized_image.shape[1])
y_coords = np.arange(0, resized_image.shape[0])
X, Y = np.meshgrid(x_coords, y_coords)

# Convert image pixel values to float for 3D plotting
image_float = resized_image.astype(float)

# Create a 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
ax.plot_surface(X, Y, image_float, cmap='gray', linewidth=0)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
ax.set_title('3D Surface Plot of Resized Image')

plt.show()