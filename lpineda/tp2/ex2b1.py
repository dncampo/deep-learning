from keras.preprocessing.image import *

train_datagen = ImageDataGenerator(
    rescale= 1. /255)

new_size = (500,500)
train_generator = train_datagen.flow_from_directory(
    '../../101_ObjectCategories',
    target_size=new_size,
    batch_size=128)