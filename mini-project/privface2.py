import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, UpSampling2D, MaxPool2D
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import glob
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from PIL import Image



filenames = glob.glob("mini-project/testimg/*.jpg")
test_imgs = []
#model face detect
face_model = cv2.CascadeClassifier('mini-project/face-detect-model.xml')


for i in range(len(filenames)):

    b_img = Image.open(filenames[i])
    b_img = b_img.convert('RGB')
    b_img = np.array(b_img)
    test_imgs.append(b_img)



generator = load_model('generator2_model.keras')
discriminator = load_model('discriminator2_model.keras')
gan = load_model('gan2_model.keras')
#generator.summary()
#discriminator.summary()
#gan.summary()


for i in range(len(test_imgs)):
    blurred_images = test_imgs[i]
    #de = test_imgs[i+18]
    #blurred_images = cv2.imread('mini-project/test3.jpg')
    #de = cv2.imread('mini-project/test3.jpg')
    gray_image = cv2.cvtColor(blurred_images, cv2.COLOR_RGB2GRAY)
    face_detect = face_model.detectMultiScale(gray_image)
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(test_imgs[i]) 

    for (x,y,w,h) in face_detect:
        #blur image
        #cv2.rectangle(blurred_images, (x,y), (x+w,y+h), (255,0,0), 2)
        #print(f"{x}, {y}, {w}, {h}")
        roi_color = blurred_images[y:y+h, x:x+w]
        
        roi_color = cv2.resize(roi_color, (128, 128))
        blurred_face = gan.predict(np.expand_dims(roi_color, axis=0))
        
        #img_h, img_w, _ = blurred_face.shape[1:]
        #blurred_images[y:y+img_h, x:x+img_w] = blurred_face[0]
        #blurred_face = cv2.resize(blurred_face[0], (img_w, img_h))
        blurred_face = cv2.resize(blurred_face[0], (w, w))
        blurred_images[y:y+h, x:x+w] = blurred_face

    plt.subplot(1, 2, 2)
    plt.title('Blurred')
    plt.imshow(blurred_images) #blur
    plt.show()