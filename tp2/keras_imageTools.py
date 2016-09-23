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
imageTrainGen=ImageDataGenerator()
train_generator=imageTrainGen.flow_from_directory(trainDir) 
imageValidGen=ImageDataGenerator()
valid_generator=imageValidGen.flow_from_directory(validDir) 

# TODO: comprobr que hace normalizacion, escalado, parches,pca

# batch 32: Test score: 8.05904769897, Test accuracy: 0.5


batch_size = 128
nb_classes = 2
nb_epoch = 20

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
