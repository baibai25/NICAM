import glob, os, gc
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, Activation 
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adamax
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

batch_size = 128
epochs = 20
class_weight = {0: 1.0, 1: 50}

# Activation Swish
def swish(x):
    return x * K.sigmoid(x)

# Build network
def build_model():
    # frames.shape = (frames width, hight, channels)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', 
                     activation=swish, input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation=swish))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(5000, activation=swish))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

if __name__ == "__main__":
    # build network
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adamax(), metrics=['acc'])
    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    #model.summary()

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        classes=['nonTC', 'TC'],
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        classes=['nonTC', 'TC'],
        class_mode='categorical',
        subset='validation'
    )
    
    print(train_generator.class_indices)
    print(len(train_generator.filenames))
    
    model.fit_generator(
        train_generator,
        #steps_per_epoch = num_train_images // batch_size,
        epochs=epochs,
        class_weight=class_weight,
        validation_data=validation_generator,
        callbacks=[es_cb]
    )

    # export model
    model.save('my_model.h5')
