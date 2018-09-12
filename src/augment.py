import glob
import keras
from keras.preprocessing.image import ImageDataGenerator

num_fakeimg = 2100665

if __name__ == "__main__":
    # Number of positive data: 71,779 
    #files = glob.glob('./data/train/TC/origin/*tif')
    #print(len(files))
    
    # Data augmentation
    gen = ImageDataGenerator(
        rotation_range=90,
        zoom_range=0.2,
        shear_range=0.2,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True,
        vertical_flip=True        
    )
    
    data_generator = gen.flow_from_directory(
        './data/train/TC',
        target_size=(64, 64),
        batch_size=1,
        class_mode=None,
        save_to_dir='./data/train/TC/fake',
        # required png or jpeg
        save_format='tif'
    )

    # Generate fake images
    i=0
    for d in data_generator:
        i+=1
        if (i == num_fakeimg):
            break

