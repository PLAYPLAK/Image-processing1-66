import cv2
import numpy as np
import math
import matplotlib.image as mpimg

#matplotlib notbook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image = cv2.imread("kmitl.jpg")
image2 = cv2.imread("chainat.jpg")

# Convert Color
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Set Image Size
height, width = 300, 500
image1 = cv2.resize(image, (width, height))
image2 = cv2.resize(image2, (width, height))

# Define the video output settings
output_file = 'blended_video.mp4'  # Replace with the desired output file name
frame_rate = 1  # Number of frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create the video writer object
video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

weight_arr = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

for w1, w2 in zip(weight_arr, weight_arr[::-1]):
    im_add = cv2.addWeighted(image1, w1, image2, w2, 0)
    video_writer.write(im_add)

for w2, w1 in zip(weight_arr, weight_arr[::-1]):
    im_add = cv2.addWeighted(image1, w1, image2, w2, 0)
    video_writer.write(im_add)


video_writer.release()
