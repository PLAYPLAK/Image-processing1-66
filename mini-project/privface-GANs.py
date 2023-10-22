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

Nmean = 0
Nstd = 1
x_train_noisy = train_img + (0.1 * np.random.normal(loc=Nmean,scale=Nstd,size=train_img.shape))
x_val_noisy = val_img + (0.1 * np.random.normal(loc=Nmean,scale=Nstd,size=val_img.shape))


def build_generator(input_shape):
    gen_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(gen_input)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    generator_model = Model(gen_input, output)

    return generator_model

def build_discriminator(input_shape):
    # Example architecture for the discriminator
    disc_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(disc_input)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output a 3-channel image
    #x = Flatten()(x)
    #output_layer = Dense(1, activation='sigmoid')(x)  # Output a single probability value

    # Create the model
    model_1 = Model(disc_input, output_layer)

    return model_1


# Create a GAN
input_shape = (128, 128, 3)  # Change this to match your image size
generator = build_generator(input_shape)
discriminator = build_discriminator(input_shape)
#generator = load_model('generator_model.keras')
#discriminator = load_model('discriminator_model.keras')

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
generator.summary()
discriminator.summary()

# Create the GAN model

discriminator.trainable = False  # Freeze the discriminator during generator training
gan_input = Input(shape=(128, 128, 3))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = Model(gan_input, gan_output)
#load gan model
#gan = load_model('gan_model.keras')
# Compile the GAN
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
gan.summary()



# Training the GAN

# Optionally, save the generated deblurred images during training here
#for epoch in range(16):


d_loss_real = discriminator.fit(train_img, train_img, batch_size=16, epochs=30)
#d_loss_fake = discriminator.fit(x_train_noisy , x_train_noisy , batch_size=16, epochs=30)
g_loss = gan.fit(train_img, train_img, batch_size=16, epochs=30)

formatted_d_loss_real = "{:.6f}".format(d_loss_real.history['loss'][0])
#formatted_d_loss_fake = "{:.6f}".format(d_loss_fake.history['loss'][0])
formatted_g_loss = "{:.6f}".format(g_loss.history['loss'][0])

print(f"\nDiscriminator Loss (Real): {formatted_d_loss_real} | Generator Loss: {formatted_g_loss}")
#print(f"\nDiscriminator Loss (Real): {formatted_d_loss_real} | Discriminator Loss (Fake): {formatted_d_loss_fake} | Generator Loss: {formatted_g_loss}")
# Save the generator model
generator.save('generator_model.keras')
# Save the discriminator model
discriminator.save('discriminator_model.keras')
# Save the GAN model
gan.save('gan_model.keras')


plt.figure(figsize=(12, 10))
plt.plot(d_loss_real.history['loss'])
#plt.plot(d_loss_fake.history['accuracy'])
plt.plot(g_loss.history['loss'])
plt.title('Loss on Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Real Loss', 'Fake Loss', 'GANs Loss'], loc='upper right')
plt.show()


for i in range(3):
    blurred_images = test_imgs[i+18]
    #de = test_imgs[i+18]
    #blurred_images = cv2.imread('mini-project/test3.jpg')
    #de = cv2.imread('mini-project/test3.jpg')
    #blurred_images = cv2.cvtColor(blurred_images, cv2.COLOR_BGR2RGB)
    #de = cv2.cvtColor(de, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(blurred_images, cv2.COLOR_RGB2GRAY)
    face_detect = face_model.detectMultiScale(gray_image)
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(test_imgs[i+18]) 

    for (x,y,w,h) in face_detect:
        #blur image
        cv2.rectangle(blurred_images, (x,y), (x+w,y+h), (255,0,0), 2)
        #print(f"{x}, {y}, {w}, {h}")
        roi_color = blurred_images[y:y+h, x:x+w]
        
        roi_color = cv2.resize(roi_color, (128, 128))
        blurred_face = gan.predict(np.expand_dims(roi_color, axis=0))
        
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
