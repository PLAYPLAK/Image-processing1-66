import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras import Model,Input
import keras.utils as image
#from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers


#Prepare Gaussian Noise Function
def add_gaussian_noise(img):
    mean = 0
    sigma = 25 
    img_noisy = img + np.random.normal(mean, sigma, img.shape)
    img_noisy = np.clip(img_noisy, 0, 255)  #กำหนกว่าค่าที่ได้จะอยู่ในช่วง 0-255
    img_noisy = img_noisy.astype('uint8')  #แปลงเป็น uint8
    
    return img_noisy

# Define parameters
Npic = 100  # Number of augmented images to generate
rotation_range = 40
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True
fill_mode = ['constant', 'nearest', 'reflect', 'wrap']


#import image
input_img = cv2.imread("lab5/kmitl.jpg")
cv2.resize((cv2.cvtColor((input_img), cv2.COLOR_BGR2RGB)), (300, 300))


output_file = 'output_video_lab5.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fps = 10.0 
frame_size = (input_img.shape[1], input_img.shape[0])  
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)


# Define ImageDataGenerator with parameters
datagen = ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    preprocessing_function=add_gaussian_noise,
    fill_mode=fill_mode[3]
)

# Creates our batch of one image
input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
pic = datagen.flow(input_img, batch_size=1)


# Randomly generate transformed images and write them to the video file
for i in range(1, Npic):
    batch = pic.next()
    im_result = batch[0].astype('uint8')
    
    # Write the transformed image to the video file
    out.write(im_result)

# Release the VideoWriter object
out.release()