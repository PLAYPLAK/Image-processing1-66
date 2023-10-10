import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,UpSampling2D,Input
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import glob
import numpy as np


filenames = glob.glob("Lab4/face_mini/*/*.jpg")
all_imgs = []

for i in range(len(filenames)):
    img = image.load_img(filenames[i],target_size=(100,100,3),interpolation="nearest")
    img = np.array(img)
    img = img/255
    all_imgs.append(img)

all_imgs = np.array(all_imgs)

train_x,test_x = train_test_split(all_imgs,random_state=32,test_size=0.3)
train_x, val_x = train_test_split(train_x, random_state=32, test_size=0.2)

noise_factor1 = 0.5
noise_factor2 = 0.1
Nmean = 0
Nstd = 1
x_train_noisy = train_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=train_x.shape))
x_val_noisy = val_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=val_x.shape))
x_test_noisy = test_x + ( noise_factor1 * np.random.normal(loc=Nmean, scale=Nstd, size=test_x.shape))
# x_test_noisy = val_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=test_x.shape))

for i in range(5):

    plt.subplot(2, 5, i+1)
    plt.imshow(x_train_noisy[i,:,:,:])
    plt.subplot(2, 5, i+6)
    plt.imshow(train_x[i,:,:,:])

plt.show()