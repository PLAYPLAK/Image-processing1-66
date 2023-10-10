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
