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
blur_imgs = []

# ... (rest of your code for loading and processing images)
face_model = cv2.CascadeClassifier('mini-project/face-detect-model.xml')

for i in range(300):
    img = image.load_img(filenames[i],target_size=(128, 128, 3), interpolation="nearest")
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = (img / 127.5) - 1
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
    b_img = tf.convert_to_tensor(b_img, dtype=tf.float32)
    b_img = (b_img / 127.5) - 1
    blur_imgs.append(b_img)


all_imgs = np.array(all_imgs)
blur_imgs = np.array(blur_imgs)

print(all_imgs.shape)
print(blur_imgs.shape)

train_img,test_img = train_test_split(all_imgs,random_state=32,test_size=0.3)
train_img,val_img = train_test_split(train_img,random_state=32,test_size=0.3)

train_blur,test_blur = train_test_split(blur_imgs,random_state=32,test_size=0.3)
train_blur,val_blur = train_test_split(train_blur,random_state=32,test_size=0.3)

def build_generator(input_shape):
    gen_input = Input(shape=(128, 128, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(gen_input)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    generator_model = Model(gen_input, output)

    return generator_model

def build_discriminator(input_shape):
    # Example architecture for the discriminator
    disc_input = Input(shape=(128, 128, 3))
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
#generator = load_model('generator_model.h5')
#discriminator = load_model('discriminator_model.h5')

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
#generator.summary()
#discriminator.summary()

# Create the GAN model
discriminator.trainable = None  # Freeze the discriminator during generator training
gan_input = Input(shape=(128, 128, 3))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = Model(gan_input, gan_output)
#load gan model
#gan = load_model('gan_model.h5')
# Compile the GAN
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
gan.summary()



# Training the GAN

# Optionally, save the generated deblurred images during training here
#for epoch in range(16):
d_loss_real = discriminator.fit(train_img, train_img, batch_size=16, epochs=1)
d_loss_fake = discriminator.fit(train_blur, train_blur, batch_size=16, epochs=1)
g_loss = gan.fit(train_img, train_img, batch_size=16, epochs=1)

formatted_d_loss_real = "{:.6f}".format(d_loss_real.history['loss'][0])
formatted_d_loss_fake = "{:.6f}".format(d_loss_fake.history['loss'][0])
formatted_g_loss = "{:.6f}".format(g_loss.history['loss'][0])

print(f"\nDiscriminator Loss (Real): {formatted_d_loss_real} | Discriminator Loss (Fake): {formatted_d_loss_fake} | Generator Loss: {formatted_g_loss}")
# Save the generator model
generator.save('generator_model.h5')
# Save the discriminator model
discriminator.save('discriminator_model.h5')
# Save the GAN model
gan.save('gan_model.h5')

predictions = gan.predict(test_img)
print(predictions.shape)

for i in range(5):
    plt.subplot(4, 5, i + 6)
    plt.imshow(test_img[i, :, :, :]) #input
    plt.subplot(4, 5, i + 11)
    plt.imshow(test_blur[i, :, :, :]) #blur
    plt.subplot(4, 5, i + 16)
    plt.imshow(predictions[i, :, :, :]) #testing images

plt.show()