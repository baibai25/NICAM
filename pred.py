import glob, os
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras import backend as K

# Activation Swish
def swish(x):
    return x * K.sigmoid(x)

if __name__ == "__main__":
   # Load data
    X_test = []
    name = []
    files = glob.glob('./test/*.tif')

    for picture in files:
        img = img_to_array(load_img(picture, grayscale=True, target_size=(64, 64)))
        X_test.append(img)
        name.append(os.path.basename(picture))
    
    X_test = np.asarray(X_test)
    X_test = X_test.astype('float32')
    X_test /= 255.0

    # Testing and export result
    model = load_model('my_model.h5', custom_objects={'swish':swish})
    y_prob = model.predict_proba(X_test)
    pd.DataFrame({'ID':name, 'prob':y_prob[:, 1]}).to_csv("submit.tsv", sep='\t', index=False, header=False)
