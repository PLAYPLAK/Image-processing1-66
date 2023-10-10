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
    img = image.load_img(filenames[i],target_size=(50,50,3),interpolation="nearest")
    img = np.array(img)
    img = img/255
    all_imgs.append(img)

all_imgs = np.array(all_imgs)

train_x,test_x = train_test_split(all_imgs,random_state=32,test_size=0.3)
train_x, val_x = train_test_split(train_x, random_state=32, test_size=0.3)

noise_factor1 = 0.5
noise_factor2 = 0.1
Nmean = 0
Nstd = 1
x_train_noisy = train_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=train_x.shape))
x_val_noisy = val_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=val_x.shape))
x_test_noisy = test_x + ( noise_factor1 * np.random.normal(loc=Nmean, scale=Nstd, size=test_x.shape))
# x_test_noisy = val_x + (noise_factor1 * np.random.normal(loc=Nmean,scale=Nstd,size=test_x.shape))

# Define the architecture of your autoencoder
input_layer = Input(shape=(50, 50, 3))  # Assuming an input shape of (50, 50, 3)
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
encoded = MaxPool2D((2, 2), padding='same')(encoded)
decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
decoded = UpSampling2D((2, 2))(decoded)
output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')


epoch =[ 2, 3, 4 ]
batch_size = [8, 16, 32]

history = autoencoder.fit (x_train_noisy, train_x,
epochs=4,
batch_size=16,
shuffle=True,
validation_data=(x_val_noisy, val_x))
# callbacks=EarlyStopping(monitor='val_loss', mode='min'))

predictions3 = autoencoder.predict(x_val_noisy)
predictions4 = autoencoder.predict(x_test_noisy)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

for i in range(5):
    plt.subplot(3, 5, i + 1)
    plt.imshow(test_x[i, :, :, :])
    plt.subplot(3, 5, i + 6)
    plt.imshow(x_test_noisy[i, :, :, :])
    plt.subplot(3, 5, i + 11)
    plt.imshow(predictions4[i, :, :, :])
plt.show()