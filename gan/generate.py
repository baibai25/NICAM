import cv2
import os
import numpy as np
import keras
from keras.models import load_model
np.random.seed(42)

# Save generated images
def save_generated_images(generated_images):

    for i in range(64):
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        image.astype(np.uint8)
        name = './generated/{}.tif'.format(str(i))
        cv2.imwrite(name, image)


# Training
def test(model_path):
    # Load model
    model = load_model(model_path)

    noise = np.random.normal(0, 1, size=(64, ) + (1, 1, 100))
    generated_images = model.predict(noise)
    
    # Output shape (64, 64, 64, 1) 
    save_generated_images(generated_images)

  
def main():
    model_path = './models/generator_epoch199.hdf5'
    test(model_path)


if __name__ == "__main__":
    main()
