import glob, os
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K

batch_size = 1

# Activation Swish
def swish(x):
    return x * K.sigmoid(x)

if __name__ == "__main__":
    # Load data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        './data/test',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False,
        seed=None,
        class_mode=None
    )

    name = []
 
    for i in range(len(path)):
        name.append(os.path.basename(path[i]))

    # Testing and export result
    model = load_model('my_model.h5', custom_objects={'swish':swish})
    y_prob = model.predict_generator(test_generator, len(test_generator.filenames), verbose=1)
    pd.DataFrame({'ID':name, 'prob':y_prob[:, 1]}).to_csv("submit.tsv", sep='\t', index=False, header=False)
