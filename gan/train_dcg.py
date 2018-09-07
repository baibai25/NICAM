import os
import numpy as np
from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization
from keras.layers import Activation, Flatten, Dropout, Reshape, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
np.random.seed(42)

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
    
    return discriminator

def gen_noise(batch_size, noise_shape):
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)


def combined_model(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model



if __name__ == "__main__":
    #noise = gen_noise(batch_size, noise_shape)
    generator = generator_model()
    generator.summary()

    discriminator = discriminator_model()
    discriminator.summary()
"""
discriminator.trainable = False
combined = combined_model(generator, discriminator)
combined.summary()


opt = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)

discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=opt)

discriminator.trainable = False
combined.compile(loss='binary_crossentropy', optimizer=opt)
"""
