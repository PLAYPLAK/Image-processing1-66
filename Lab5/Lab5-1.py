import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras import Model,Input
import keras.utils as image
#from keras.wrappers.scikit_learn import KerasRegressor
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras import optimizers

#from tensorflow.keras.datasets import fashion_mnist
#from sklearn.model_selection import train_test_split

input_img = cv2.imread("lab5/Grid.png")

#Define resize factor
Reduce_factors = [2, 4, 8]


#Define interpolation method
inter_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
text = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"]

for l in range (4):
    Scale_factor = 1/Reduce_factors[0]
    image_resize = cv2.resize(input_img, None, fx=Scale_factor, fy=Scale_factor,interpolation=inter_methods[l])     
    plt.subplot(3,4,l+1)
    plt.title(f"{Reduce_factors[0]}-{text[l]}")
    plt.imshow(image_resize)

for l in range (4):
    Scale_factor = 1/Reduce_factors[1]
    image_resize = cv2.resize(input_img, None, fx=Scale_factor, fy=Scale_factor,interpolation=inter_methods[l])     
    plt.subplot(3,4,l+5)
    plt.title(f"{Reduce_factors[1]}-{text[l]}")
    plt.imshow(image_resize)

for l in range (4):
    Scale_factor = 1/Reduce_factors[2]
    image_resize = cv2.resize(input_img, None, fx=Scale_factor, fy=Scale_factor,interpolation=inter_methods[l])     
    plt.subplot(3,4,l+9)
    plt.title(f"{Reduce_factors[2]}-{text[l]}")
    plt.imshow(image_resize)

plt.show()