import glob, os, gc
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, Activation, BatchNormalization, Input 
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adamax, Adam, SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model

batch_size = 128
epochs = 10
class_weight = {0:1.0, 1: 3.0}
#model_path = './tmp/weights-improvement-20-0.95.hdf5'

# Activation Swish
def swish(x):
    return x * K.sigmoid(x)

# Build network
def build_model():
    # frames.shape = (frames width, hight, channels)

    resnet = VGG16(include_top=False, input_shape=(64, 64, 3), weights='imagenet')
    input_tensor = Input(shape=resnet.output_shape[1:])
    
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnet.output_shape[1: ]))
    top_model.add(Dense(256, activation=swish))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    
    model = Model(input=resnet.input, output=top_model(resnet.output))
    
    for layer in model.layers[:15]:
        layer.trainable = False        

    return model

if __name__ == "__main__":
    # build network
    model = build_model()
    model.summary()
    #model.load_weights(model_path)
 
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['acc'])
    es_cb = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    tb_cb = TensorBoard(log_dir='./logs')
    cp_path = './tmp/{epoch:02d}-{val_loss:.2f}.hdf5' 
    cp_cb = ModelCheckpoint(filepath=cp_path, verbose=1)
    #model.summary()

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(64, 64),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        classes=['nonTC', 'TC'],
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(64, 64),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        classes=['nonTC', 'TC'],
        class_mode='binary',
        subset='validation'
    )
    
    print(train_generator.class_indices)
  
    model.fit_generator(
        train_generator,
        epochs=epochs,
        class_weight=class_weight,
        validation_data=validation_generator,
        callbacks=[tb_cb, cp_cb]
    )

    # export model
    model.save('my_model.h5')
