import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal

img = cv2.imread('lab3/kmitl.jpg')

img = cv2.resize(img,(224,224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = expand_dims(img,axis=0)


print(img.shape)

img_mean = [123.68,116.779,103.939]
img_result = img - img_mean


plt.imshow(img_result[0])
plt.show()


