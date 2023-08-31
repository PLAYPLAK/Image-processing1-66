import cv2
import numpy as np
import math
import matplotlib.image as mpimg

#matplotlib notbook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
############################################
fig = plt.figure(figsize=(10, 5))
rows = 1
columns = 2


image = cv2.imread("lab1/kmitl.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
reduced_bit_depth = 4

if(reduced_bit_depth>=0):
    quantization_levels = 2 ** reduced_bit_depth
    scaling_factor = 255 / (quantization_levels - 1)

    # Perform quantization on each color channel separately
    quantized_image = np.round(gray_image / scaling_factor) * scaling_factor

    fig.add_subplot(rows, columns, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('on')
    plt.title("Defual image")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(quantized_image, cmap='gray')
    plt.axis('on')
    plt.title(f"Quantized 8 -> {reduced_bit_depth} Bit")
    

plt.show()