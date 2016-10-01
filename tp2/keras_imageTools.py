import os, random, math
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
#from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Flatten
from scipy import io
import time

# airplanes vs motorbikes
trainDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/airplanesVSmotorbikes_train/'
validDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/airplanesVSmotorbikes_valid/'

augmentationv=[0,1]
nhidv=[32,64,128,256]
nhid2v=[0,32,64]
dropoutv=[0,0.1,0.2,.5]
actfuncv=['relu','sigmoid']
results=[['augmentation','nhid','nhid2','dropoout','actfunc','trainacc','testacc']]
e=0
E=len(augmentationv)*len(nhidv)*len(nhid2v)*len(dropoutv)*len(actfuncv)
start=time.time()
fout=open('/home/leandro/tmp/keras.mat','bw')
for augmentation in augmentationv[::-1]:
    for nhid in nhidv[::-1]:
        for nhid2 in nhid2v[::-1]:
            for dropout in dropoutv[::-1]:
                for actfunc in actfuncv:
                    etime=time.time()-start
                    eta=-1
                    if e>0:
                        eta=etime/e*(E-e)/3600
                        
                    print('%d/%d - %0.2fs (ETA: %0.2fh)' %(e,E,etime,eta))
                    if augmentation:
                        imageGenTrain=ImageDataGenerator(rotation_range=40,
                                                     rescale=1./255,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     shear_range=0.2,
                                                     zoom_range=0.2,
                                                         horizontal_flip=True
                                                         )
                    else:
                        imageGenTrain=ImageDataGenerator(rescale=1./255)
                    imageGenTest=ImageDataGenerator(rescale=1./255)
    
                    imsize=64
                    batch_size=64
                    nb_classes = 2
                    nb_epoch = 10 

                    train_generator=imageGenTrain.flow_from_directory(trainDir,target_size=(imsize, imsize),batch_size=batch_size, class_mode='binary')
                
                    valid_generator=imageGenTest.flow_from_directory(validDir,target_size=(imsize, imsize),batch_size=batch_size, class_mode='binary')

                    trainSamples=train_generator.N
                    validationSamples=valid_generator.N

                    model = Sequential()
                    model.add(Flatten(input_shape=(imsize,imsize,3))) 
                    model.add(Dense(nhid))
                    model.add(Activation(actfunc))
                    model.add(Dropout(dropout))
                    if nhid2>0:
                        model.add(Dense(nhid2))
                        model.add(Activation(actfunc))
                        model.add(Dropout(dropout))
                        
                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))
                    #model.summary()

                    model.compile(loss='binary_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])

                    history = model.fit_generator(train_generator,
                                                  samples_per_epoch=batch_size*20,
                                                  nb_epoch=nb_epoch,verbose=1)
                    score = model.evaluate_generator(valid_generator,
                                                     validationSamples,verbose=0)

                    #model.save('keras_image.h5')
                    results.append([augmentation,nhid,dropout,actfunc,history.history['acc'][-1],score[1]])
                    restitle='results%d' %e
                    print(restitle)
                    
                    io.savemat(fout,{restitle: results[-1]})
                    e+=1
#print(results)
