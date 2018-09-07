import glob, os, gc
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, Activation, BatchNormalization 
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adamax, Adam, SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

batch_size = 128
epochs = 20
class_weight = {0:1.0, 1: 3.0}

# Activation Swish
def swish(x):
    return x * K.sigmoid(x)

# Build network
def build_model():
    # frames.shape = (frames width, hight, channels)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
         data_format='channels_last', input_shape=(64, 64, 1)))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(256, activation=swish))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

if __name__ == "__main__":
    # build network
    model = build_model()
    model.compile(loss='binary_crossentropy',
            optimizer=SGD(momentum=1e-4, decay=0.9, nesterov=True), metrics=['acc'])
    es_cb = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    tb_cb = TensorBoard(log_dir='./logs')
    cp_path = 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5' 
    cp_cb = ModelCheckpoint(filepath=cp_path, verbose=1)
    model.summary()
 
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        classes=['nonTC', 'TC'],
        class_mode='binary',
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
        class_mode='binary',
        subset='validation'
    )
    
    print(train_generator.class_indices)
    #print(len(train_generator.filenames))
  
    model.fit_generator(
        train_generator,
        epochs=epochs,
        class_weight=class_weight,
        validation_data=validation_generator,
        callbacks=[tb_cb, cp_cb]
    )

    # export model
    model.save('my_model.h5')

