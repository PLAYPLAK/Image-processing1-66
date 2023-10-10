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

for i in range(500):
    img = image.load_img(filenames[i],target_size=(100,100,3),interpolation="nearest")
    img = np.array(img)
    img = img/255
    all_imgs.append(img)

all_imgs = np.array(all_imgs)

train_x,test_x = train_test_split(all_imgs,random_state=32,test_size=0.3)
train_x, val_x = train_test_split(train_x, random_state=32, test_size=0.2)

#add noisy
noise_factor1 = 0.5
noise_factor2 = 0.1
Nmean = 0
Nstd = 1
x_train_noisy = train_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=train_x.shape))
x_val_noisy = val_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=val_x.shape))
x_test_noisy = test_x + ( noise_factor1 * np.random.normal(loc=Nmean, scale=Nstd, size=test_x.shape))
# x_test_noisy = val_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=test_x.shape))


#Encideed
Input_img = Input(shape=(100, 100, 3))
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x3 = MaxPool2D( (2, 2))(x2)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)

#decodeed
x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x5 = UpSampling2D((2, 2))(x4)
x6 = Conv2D(128, (3, 3), activation='relu', padding='same')(x5)
x7 = Conv2D(256, (3, 3), activation='relu', padding='same')(x6)
decoded = Conv2D(3, (3, 3), padding='same')(x7)
autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

callback = EarlyStopping(monitor='loss', patience=3)
history = autoencoder.fit (x_train_noisy, train_x,
epochs=4,
batch_size=32,
shuffle=True,
validation_data=(x_val_noisy, val_x),
callbacks=[callback],verbose=1)

#predictions1 = autoencoder.predict(x_val_noisy)
predictions2 = autoencoder.predict(x_test_noisy)

plt.figure(figsize=(12, 10))

plt.subplot(6, 8, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

for i in range(5):
    plt.subplot(4, 5, i + 6)
    plt.imshow(test_x[i, :, :, :])
    plt.subplot(4, 5, i + 11)
    plt.imshow(x_test_noisy[i, :, :, :])
    plt.subplot(4, 5, i + 16)
    plt.imshow(predictions2[i, :, :, :])

plt.show()