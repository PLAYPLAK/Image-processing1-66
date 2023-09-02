import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal

img = cv2.imread('test.jpg')
img = cv2.resize(img,(224,224))
img = expand_dims(img,axis=0)
img_mean = [103.939,116.779,123.68]
img = img - img_mean


#4.3
model = VGG16()
img = img[0]
img_result = img.copy()
kernels, biases = model.layers[1].get_weights()
image_sum = np.zeros((224,224,3))
for i in range(64):
    img_result[ :, :, 0] = signal.convolve2d(img[ :, :, 0], kernels[:, :, 0, i], mode='same',boundary='fill', fillvalue=0)
    img_result[ :, :, 1] = signal.convolve2d(img[ :, :, 1], kernels[:, :, 1, i], mode='same',boundary='fill', fillvalue=0)
    img_result[ :, :, 2] = signal.convolve2d(img[ :, :, 2], kernels[:, :, 2, i], mode='same',boundary='fill', fillvalue=0)
    image_sum = img_result[ :, :, 0] + img_result[ :, :, 1] + img_result[ :, :, 2]

    # plt.imshow(img_result[0][ :, :, 0])
    plt.subplot(8, 8, i + 1)
    for i,val in enumerate(image_sum):
        for x,val2 in enumerate(val):
            if val2 < 0:
                image_sum[i][x] = 0

    plt.imshow(image_sum, cmap='jet')
plt.show()
