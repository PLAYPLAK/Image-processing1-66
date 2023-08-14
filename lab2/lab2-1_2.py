import cv2
import numpy as np

from matplotlib import pyplot as plt

# Image pixel adjustment
image = cv2.imread("lab1/kmitl.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image)

height, width = 300, 500
image = cv2.resize(image, (width, height))
# Define the video output settings
output_file = 'Gamma_video.mp4'  # Replace with the desired output file name
frame_rate = 1  # Number of frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))


y_low = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y_High = [1.5, 2, 2.5, 3, 3.5, 5.0, 5.5, 10, 15, 20]

for y in y_low:
    
    a = 1
    b = 0
    image_gamma = image**y
    output_image = (a * image_gamma ) + b
    output_image = np.clip(output_image, a_min=0, a_max=255).astype(np.uint8)
    video_writer.write(output_image)

for y in y_High :
    
    a = 1
    b = 0
    image_gamma = image**y
    output_image = (a * image_gamma) + b
    output_image = np.clip(output_image, a_min=0, a_max=255).astype(np.uint8)
    video_writer.write(output_image)

video_writer.release()
