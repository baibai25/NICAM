import glob, os
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img

if __name__ == "__main__":
    
    X = []
    y = []
    
    files = glob.glob('./train/TC/*.tif')

    print('Load dataset')
    for picture in files:
        img = img_to_array(load_img(picture, target_size=(64, 64)))
        X.append(img)
        y.append(0)

    print('Convert data')
    X = np.asarray(X)
    y = np.asarray(y)
    X = X.astype('float32')
    X /= 255.0

    print(X.shape)
