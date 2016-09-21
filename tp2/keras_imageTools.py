import os, random, math, pandas
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
#plt.close('all')

# airplanes vs motorbikes
dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/airplanesVSmotorbikes/'

# cargador por defecto. Cada subfolder es una clase.
imageTrainGen=keras.preprocessing.image.ImageDataGenerator
imageTrainGen.flow_from_directory(dataDir) # se pueden hacer directorios diferentes para train/test

# TODO: comprobr que hace normalizacion, escalado, parches,pca

batch_size = 32
nb_classes = 2
nb_epoch = 20

model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(imageTrainGen,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                              verbose=1, validation_data=imageTrainGen)
score = model.evaluate_generator(imageTrainGen, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])
