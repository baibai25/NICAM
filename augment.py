import glob
import keras
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    # Number of positive data 
    #files = glob.glob('./data/train/TC/origin/*tif')
    #print(len(files))
    
    # Data augmentation
    gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True        
    )
    
    data_generator = gen.flow_from_directory(
        './data/train/TC',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=batch_size,
        save_to_dir='./data/train/TC/fake',
        save_format='tif'
    )
   
    # Generate fake images
    for i in range(2):
        data_generator.next()


