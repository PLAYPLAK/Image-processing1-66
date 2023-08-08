import cv2
import numpy as np

from matplotlib import pyplot as plt

# Image pixel adjustment
image = cv2.imread("lab1/kmitl.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

a = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
b = [10, 20, 30, 40, 50]

# Define the video output settings
output_file = 'liner_video.mp4'  # Replace with the desired output file name
frame_rate = 1  # Number of frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (image.shape[1], image.shape[0]))


for num in a :
    for y in range(image.shape[0]) :
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                pixel = image[y, x, c]
                cal_value = int(num * pixel + b[0])
                
                #output_pixel = max(0, min(255, cal_value))

#combline_image = np.hstack((image, output_pixel))

plt.imshow(combline_image)
plt.show()
#video_writer.write(combline_image)


#video_writer.release()