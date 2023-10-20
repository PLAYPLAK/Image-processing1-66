import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import glob
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


filenames = glob.glob("mini-project/archive/images/*.png")
all_imgs = []
blur_imgs = []
#model face detect
face_model = cv2.CascadeClassifier('mini-project/face-detect-model.xml')

for i in range(400):
    img = image.load_img(filenames[i],target_size=(128, 128, 3), interpolation="nearest")
    img = np.array(img)
    img = img/255
    all_imgs.append(img)

    b_img = Image.open(filenames[i])
    b_img = b_img.convert('RGB')
    b_img = np.array(b_img)
    

    gray_image = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)
    face_detect = face_model.detectMultiScale(gray_image)
    for (x,y,w,h) in face_detect:
        #blur image
        roi_color = b_img[y:y+h, x:x+w]
        roi_b = cv2.GaussianBlur(roi_color, (23,23), 30)
        b_img[y:y+h, x:x+w] = roi_b

    b_img = cv2.resize(b_img,(128, 128))
    #b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
    b_img = b_img/255
    blur_imgs.append(b_img)


all_imgs = np.array(all_imgs)
blur_imgs = np.array(blur_imgs)


train_img,test_img = train_test_split(all_imgs,random_state=32,test_size=0.3)
train_img,val_img = train_test_split(train_img,random_state=32,test_size=0.3)

train_blur,test_blur = train_test_split(blur_imgs,random_state=32,test_size=0.3)
train_blur,val_blur = train_test_split(train_blur,random_state=32,test_size=0.3)


#Encoder
Input_img = Input(shape=(128, 128, 3))
x = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D( (2, 2))(x)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

#decoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), padding='same')(x)

#autoencoder = Model(Input_img, decoded)
autoencoder = load_model('face-blur-autoencoder_model.h5')
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

callback = EarlyStopping(monitor='loss', patience=3)
history = autoencoder.fit (train_img, train_blur,
        epochs=1,
        batch_size=16,
        shuffle=True,
        validation_data=(val_img, val_blur),
        callbacks=[callback],verbose=1)

autoencoder.save('face-blur-autoencoder_model.h5')

#predictions = autoencoder.predict(test_img)
predictions = autoencoder.predict(test_img)

plt.figure(figsize=(12, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

for i in range(5):

    plt.subplot(4, 5, i + 6)
    plt.imshow(test_img[i, :, :, :]) #input
    plt.subplot(4, 5, i + 11)
    plt.imshow(test_blur[i, :, :, :]) #blur
    plt.subplot(4, 5, i + 16)
    plt.imshow(predictions[i, :, :, :]) #testing images

plt.show()

""""""