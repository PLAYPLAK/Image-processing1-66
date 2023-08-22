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
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

model = VGG16()
print(model.summary())

kernels, biases = model.layers[1].get_weights()
print(model.layers[1].get_config())
print(kernels[:,:,:,0])
print(kernels.shape)
img = img_to_array(img)
img = cv2.resize(img,(224,224))

img = expand_dims(img,axis=0)
img_ready = preprocess_input(img)

model = Model(inputs=model.inputs,outputs=model.layers[1].output)
print(model.summary())
#
feature_maps = model.predict(img_ready)
print(feature_maps)
for i in range(64):
    plt.subplot(8,8,i+1);plt.imshow(feature_maps[0][:,:,i],cmap='jet')
plt.title('4.1')
plt.show()