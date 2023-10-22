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
test_imgs = []
#model face detect
face_model = cv2.CascadeClassifier('mini-project/face-detect-model.xml')



for i in range(len(filenames)):
    img = image.load_img(filenames[i],target_size=(128, 128, 3), interpolation="nearest")
    img = np.array(img)
    img = img/255
    all_imgs.append(img)

    if len(all_imgs) > (70/100) * len(filenames):
        b_img = Image.open(filenames[i])
        b_img = b_img.convert('RGB')
        b_img = np.array(b_img)
        test_imgs.append(b_img)



all_imgs = np.array(all_imgs)

train_img, _  = train_test_split(all_imgs,random_state=32,test_size=0.3)
train_img,val_img = train_test_split(train_img,random_state=32,test_size=0.3)




#Encoder
Input_img = Input(shape=(128, 128, 3))
x = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D( (2, 2))(x)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

#Decoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), padding='same')(x)

#autoencoder = Model(Input_img, decoded)
autoencoder = load_model('face-blur-autoencoder_model.keras')
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

"""
callback = EarlyStopping(monitor='loss', patience=3)
history = autoencoder.fit (train_img, train_img,
        epochs=30,
        batch_size=16,
        shuffle=True,
        validation_data=(val_img, val_img),
        callbacks=[callback],verbose=1)


autoencoder.save('face-blur-autoencoder_model.keras')


plt.figure(figsize=(12, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()


"""
for i in range(3):
    blurred_images = test_imgs[i+10]
    #de = test_imgs[i+18]
    #blurred_images = cv2.imread('mini-project/test3.jpg')
    #de = cv2.imread('mini-project/test3.jpg')
    #blurred_images = cv2.cvtColor(blurred_images, cv2.COLOR_BGR2RGB)
    #de = cv2.cvtColor(de, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(test_imgs[i+10])
    
    gray_image = cv2.cvtColor(blurred_images, cv2.COLOR_RGB2GRAY)
    face_detect = face_model.detectMultiScale(gray_image)

    for (x,y,w,h) in face_detect:
        #blur image
        cv2.rectangle(blurred_images, (x,y), (x+w,y+h), (255,0,0), 2)
        #print(f"{x}, {y}, {w}, {h}")
        roi_color = blurred_images[y:y+h, x:x+w]
        
        roi_color = cv2.resize(roi_color, (128, 128))
        blurred_face = autoencoder.predict(np.expand_dims(roi_color, axis=0))
        
        img_h, img_w, _ = roi_color.shape
        blurred_face = cv2.resize(blurred_face[0], (img_w, img_h))
        print('----------')
        print(blurred_images[y:y+img_h, x:x+img_w].shape)
        print('----------')
        print(blurred_face.shape)
        print('----------')
        blurred_images[y:y+img_h, x:x+img_w] = cv2.GaussianBlur(blurred_face, (23,23), 30)
    
    #original_height, original_width = blurred_images.shape[:2]
    #de = cv2.resize(de, (original_height*0.5, original_width*0.5))
    #blurred_images = cv2.resize(blurred_images, (original_height*0.5, original_width*0.5))
    
    plt.subplot(1, 2, 2)
    plt.title('Blurred')
    plt.imshow(blurred_images) #blur
    
    plt.show()
