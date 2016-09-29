import os, random, math, pandas
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Reshape,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
#plt.close('all')

# airplanes vs motorbikes
trainDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/airplanesVSmotorbikes_train/'
validDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/airplanesVSmotorbikes_valid/'

# cargador por defecto. Cada subfolder es una clase.
#imageGen=ImageDataGenerator(featurewise_std_normalization=True,  featurewise_center=True)
imageGenTrain=ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
                                 fill_mode='nearest')
imageGenTest=ImageDataGenerator(rescale=1./255)

imsize=128
batch_size = 128
nb_classes = 2
nb_epoch = 20

train_generator=imageGenTrain.flow_from_directory(trainDir,target_size=(imsize, imsize),batch_size=batch_size, class_mode='binary')
#,save_to_dir='/home/leandro/tmp/')

valid_generator=imageGenTest.flow_from_directory(validDir,target_size=(imsize, imsize),batch_size=batch_size, class_mode='binary')

# train_generator=imageGenTrain.flow_from_directory(trainDir,target_size=(imsize, imsize)) # categorical labels




trainSamples=train_generator.N
validationSamples=valid_generator.N

model = Sequential()


# esto creo que no anda
#model.add(Reshape((imsize*imsize*3,), input_shape=(imsize,imsize,3)))

model.add(Convolution2D(32, 3, 3, input_shape=(imsize, imsize,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 
#model.add(Flatten(input_shape=(imsize,imsize,3))) 

model.add(Dense(100))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2))
# model.add(Activation('softmax'))

model.add(Dense(1))
model.add(Activation('sigmoid'))



model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    samples_per_epoch=batch_size*20, nb_epoch=nb_epoch,
                              verbose=1)
# history = model.fit_generator(train_generator,
#                     samples_per_epoch=batch_size, nb_epoch=nb_epoch,
#                               verbose=1, validation_data=train_generator,nb_val_samples=batch_size)
score = model.evaluate_generator(valid_generator,validationSamples, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])
#model.save('keras_image.h5')
