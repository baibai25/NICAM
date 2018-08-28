import glob, os, gc
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Preprocessing data
def pre_data(X, y):
    # split and convert data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    return X_train, X_val, y_train, y_val

# Activation Swish
def swish(x):
    return x * K.sigmoid(x)

# Build network
def build_model():
    # frames.shape = (frames width, hight, channels)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', 
                     activation=swish, input_shape=(64, 64, 3)))
    model.add(Conv2D(32, (3, 3), activation=swish))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation=swish))
    model.add(Conv2D(64, (3, 3), activation=swish))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation=swish))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model


if __name__ == "__main__":
    # Load data
    X = []
    y = []
    tc = glob.glob('./train/TC/*.tif')
    nontc = glob.glob('./train/nonTC/*.tif')

    print('Load dataset')
    for picture in tc:
        img = img_to_array(load_img(picture, target_size=(64, 64)))
        X.append(img)
        y.append(1)

    for picture in nontc:
        img = img_to_array(load_img(picture, target_size=(64, 64)))
        X.append(img)
        y.append(0)

    print('Convert data')
    X = np.asarray(X)
    y = np.asarray(y)
    X = X.astype('float32')
    X /= 255.0

    X_train, X_val, y_train, y_val = pre_data(X, y)
    del X
    del y
    gc.collect()
    print(X_train.shape)
    print(y_train.shape)

"""
    # Learning
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adamax(), metrics=['acc'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, 
            validation_data=(X_val, y_val), verbose=1, shuffle=True)
    
    # export model
    model.save('my_model.h5')
"""
