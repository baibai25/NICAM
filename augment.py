import glob
import keras
from keras.preprocessing.image import ImageDataGenerator

batch_size =128

if __name__ == "__main__":
    # Number of positive data: 71,779 
    #files = glob.glob('./data/train/TC/origin/*tif')
    #print(len(files))
    
    # Data augmentation
    gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
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
        class_mode=None,
        save_to_dir='./data/train/TC/fake',
        save_format='tif'
    )
"""   
    # Generate fake images
    for i in range(30):
        data_generator.next()
"""
