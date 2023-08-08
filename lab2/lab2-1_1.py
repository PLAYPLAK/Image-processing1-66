import cv2
import numpy as np

from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 10))
  
# setting values to rows and column variables
rows = 2
columns = 21

# Image pixel adjustment
image = cv2.imread("lab1/kmitl.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image)

height, width = 300, 500
image = cv2.resize(image, (width, height))
# Define the video output settings
output_file = 'lener_video.mp4'  # Replace with the desired output file name
frame_rate = 1  # Number of frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

print(image.shape)

n = 1
for a in range(-10, 11) :
    print(a)
    for b in range(0, 10) :
        image[:, :, 0] = (1 * image[:, :, 0] ) + b

    fig.add_subplot(rows, columns, n)
    plt.imshow(image)
    # video_writer.write(image)
    n = n + 1


n2 = 22
for a in range(-10, 11) :
    print(a)
    for b in range(0, 10) :
        image[:, :, 1] = (20 * image[:, :, 1] ) + b

    fig.add_subplot(rows, columns, n2)
    plt.imshow(image)
    # video_writer.write(image)
    n2 = n2 + 1

#print(image)

#  video_writer.release()
plt.show()


# combline_image = np.hstack((image, r_chanel))