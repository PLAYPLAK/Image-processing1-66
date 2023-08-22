import cv2
import numpy as np

from matplotlib import pyplot as plt

# Image pixel adjustment
image = cv2.imread("lab2/kmitl.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image)

height, width = 300, 500
image = cv2.resize(image, (width, height))
# Define the video output settings
output_file = 'Gamma_video.mp4'  # Replace with the desired output file name
frame_rate = 1  # Number of frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

quantization_levels = 2 ** 8

y_low = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y_High = [1.5, 2, 2.5, 3, 3.5, 5.0, 5.5, 10, 15, 20]
image =image.astype(np.float64)

# Gamma 0 < (y) < 1
for y in y_low:
    
    a = 1
    b = 0
    image_gamma = image**y
    output_image = (a * image_gamma ) + b

    quantized_image = ((output_image - np.min(output_image))/(np.max(output_image) - np.min(output_image))) * ((2**8)-1)
    quantized_image = quantized_image.astype(np.uint8)
    video_writer.write(quantized_image)
# Gamma (y) > 1
for y in y_High :
    
    a = 1
    b = 0
    image_gamma = image**y
    output_image = (a * image_gamma) + b
    
    quantized_image = ((output_image - np.min(output_image))/(np.max(output_image) - np.min(output_image))) * ((2**8)-1)
    quantized_image = quantized_image.astype(np.uint8)
    video_writer.write(quantized_image)

video_writer.release()
