#!/usr/bin/python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation

model= Sequential([ Dense(32,input_dim=784),Activation('relu'),Dense(10),Activation('softmax')]) # NN normal de 32 unidades ocultas, 784 feat y 10 salidas
# model.add(Dense(Nunits,ninput))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) # binario. metrics son otras medidas que se pueden usar en algunos casos de la optimización.
#model.compile(optimizer='rmsprop',loss='mse') # regresión

# descenso por gradiente estocástico 
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9) # se pueden definir módulos arbitrarios  
model.compile(loss='mean_squared_error',optimizer=sgd)

import numpy as np
data=np.random.random((1000,784))
labels=np.random.randint(2,size=(1000,1))
model.fit(data,labels,nb_epoch=150,batch_size=32)

testdata=np.random.random((1000,784))
testlabels=np.random.randint(2,size=(1000,1))

print model.evaluate(test_data,test_labels,nb_epoch=10,batchsize=32)
predictions = model.predict(test_data)

