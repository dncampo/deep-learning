import os, random, math, pandas
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Reshape
#plt.close('all')

# airplanes vs motorbikes
trainDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/airplanesVSmotorbikes_train/'
validDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/airplanesVSmotorbikes_valid/'

# cargador por defecto. Cada subfolder es una clase.
#imageGen=ImageDataGenerator(featurewise_std_normalization=True,  featurewise_center=True)
imageGen=ImageDataGenerator()

train_generator=imageGen.flow_from_directory(trainDir) 

valid_generator=imageGen.flow_from_directory(validDir) 

# TODO: como hacer normalización desde el flow_directory??

# batch 32 y 128: Test score: 8.05904769897, Test accuracy: 0.5


batch_size = 32
nb_classes = 2
nb_epoch = 5

trainSamples=train_generator.N
validationSamples=valid_generator.N

model = Sequential()

#model.add(Dense(512, input_shape=(784,)))
# O hago un reshape del input, o uso otro tipo de capa...
# model.add(Dense(512, input_shape=(256,256,3,)))
model.add(Reshape((256*256*3,), input_shape=(256,256,3)))
model.add(Dense(100, input_shape=(256*256*3,)))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(nb_classes)) 
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    samples_per_epoch=batch_size, nb_epoch=nb_epoch,
                              verbose=1)
# history = model.fit_generator(train_generator,
#                     samples_per_epoch=batch_size, nb_epoch=nb_epoch,
#                               verbose=1, validation_data=train_generator)
score = model.evaluate_generator(valid_generator,validationSamples, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('keras_image.h5')
