import os, sys
import tifffile
import numpy as np
import keras
from keras.models import load_model
np.random.seed(42)

# Save generated images as 32bit grayscale tiff
def save_generated_images(generated_images):

    for i in range(len(generated_images)):
        image = generated_images[i, :, :, :]
        #image += 1
        #image *= 127.5
        #image.astype(np.uint8)
        min = image.min()
        max = image.max()
        image = (image - min) / (max - min)
        name = './generated/{}.tif'.format(str(i))
        tifffile.imsave(name, image)


# Training
def test(model_path, generate_size):
    # Load model
    model = load_model(model_path)
    
    # Generate images
    noise = np.random.normal(0, 1, size=(generate_size, ) + (1, 1, 100))
    generated_images = model.predict(noise)
    
    # Output shape (generate_size, 64, 64, 1) 
    save_generated_images(generated_images)

  
def main():
    model_path = './models/generator_epoch3.hdf5'
    generate_size = 64
    test(model_path, generate_size)


if __name__ == "__main__":
    main()
