from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
np.random.seed(42)

def load_dataset(dataset_path, batch_size):
    
    gen = ImageDataGenerator()
    
    data_generator = gen.flow_from_directory(
        dataset_path,
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False,
        seed=None,
        class_mode=None,
    )

    return data_generator

# define model
def generator_model():
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    generator = Sequential()
    generator.add(Dense(4*4*512, input_shape=(1, 1, 100), kernel_initializer=kernel_init))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((4, 4, 512)))

    generator.add(Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    
    generator.add(Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    
    generator.add(Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    
    generator.add(Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init))
    generator.add(Conv2D(1, kernel_size=(4, 4), padding='same', strides=(1, 1)))
    generator.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    return generator

def discriminator_model():
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'   
    
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(4, 4), padding='same', strides = (2,2),
        input_shape=(64, 64, 1), kernel_initializer=kernel_init))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, kernel_size=(4, 4), padding='same', strides=(2,2), kernel_initializer=kernel_init))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(256, kernel_size=(4, 4), padding='same', strides=(2,2), kernel_initializer=kernel_init))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(512, kernel_size=(4, 4), padding='same', strides=(2,2), kernel_initializer=kernel_init))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))
    
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    return discriminator

# Save generated images
def save_generated_images(generated_images, epoch, batch_number):
    
    new_dir = './output/epoch-{}'.format(epoch)
    os.mkdir(new_dir)

    for i in range(64):
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        image.astype(np.uint8)
        #image = np.squeeze(image)
        name = './output/epoch-{}/{}.tif'.format(str(epoch), str(i))
        cv2.imwrite(name, image)

def train(dataset_path, batch_size, epochs):
    # Build network
    generator = generator_model()
    discriminator = discriminator_model()
    discriminator.trainable = False
    
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)
    #discriminator.summary()
    #generator.summary()
    #gan.summary()

    # Load dataset
    dataset_generator = load_dataset(dataset_path, batch_size)
    sample_size = int(len(dataset_generator.filenames))
    number_of_batches = int(len(dataset_generator.filenames) / batch_size)
    #71779

    # plot variables
    adversarial_loss = np.empty(shape=1)
    discriminator_loss = np.empty(shape=1)
    batches = np.empty(shape=1)

    current_batch = 0

    # Train
    for epoch in range(epochs):
        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " :")

        # Train discriminator
        for batch_number in range(number_of_batches):
            # Rescale images between -1 and 1
            real_images = dataset_generator.next()
            real_images /= 127.5
            real_images -= 1

            current_batch_size = real_images.shape[0]

            # Generate noise and image
            # noise.shape = (current_batch_size, 1, 1, 100)
            noise = np.random.normal(0, 1, size=(current_batch_size, ) + (1, 1, 100))
            generated_images = generator.predict(noise)

            # Add noise to the labels
            real_y = (np.ones(current_batch_size) - np.random.random_sample(current_batch_size) * 0.2)
            fake_y = np.random.random_sample(current_batch_size) * 0.2

    
            # Train discriminator
            # learn real_y and fake_y
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(real_images, real_y)
            d_loss += discriminator.train_on_batch(generated_images, fake_y)
            discriminator_loss = np.append(discriminator_loss, d_loss)


            # Train generator
            # try to mislead the discriminator by giving fake lavel
            discriminator.trainable = False
            noise = np.random.normal(0, 1, size=(current_batch_size * 2,) + (1, 1, 100))
            fake_y = (np.ones(current_batch_size * 2) - np.random.random_sample(current_batch_size * 2) * 0.2)
            g_loss = gan.train_on_batch(noise, fake_y)
            adversarial_loss = np.append(adversarial_loss, g_loss)
            batches = np.append(batches, current_batch)

            print('Step: {}'.format(batch_number))
            #(64x64x64x1) 
       
            # Each 50 batches show and save images
            #if((batch_number + 1) % 50 == 0 and current_batch_size == batch_size):
                #save_generated_images(generated_images, epoch, batch_number)

            current_batch += 1
       
        save_generated_images(generated_images, epoch, batch_number)
  
        # Save the model weights each 5 epochs 
        if (epoch + 1) % 5 == 0:
            discriminator.trainable = True
            generator.save('./models/generator_epoch' + str(epoch) + '.hdf5')
            discriminator.save('./models/discriminator_epoch' + str(epoch) + '.hdf5')

def main():
    dataset_path = './data'
    batch_size = 64
    epochs = 10
    train(dataset_path, batch_size, epochs)


if __name__ == "__main__":
    main()
