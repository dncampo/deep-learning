#!/usr/bin/python
# -*- coding: utf-8 -*-
import theano
import numpy as np
from logRegression import logRegression
rng = np.random

# TP2.1.a: 
# Revise​este tutorial de theano​ y resuelva el ejercicio propuesto. 
a = theano.tensor.vector() 
out = a + a ** 10          
f = theano.function([a], out)
#print(f([0, 1, 2]))

# Modify and execute this code to compute this expression: 
# a ** 2 + b ** 2 + 2 * a * b.
b = theano.tensor.vector() 
out = a**2 + b**2 + 2* a * b          
g = theano.function([a,b], out)
#print(g([0, 1, 2],[0, 2, 3]))

# TP2.1.b:
# Modifique​este ejemplo de regresión logística​ para: 
# generate a dataset: D = (input_values, target_class)
N = 400               
feats = 784           
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
nsteps = 300
print 'Batch training:'
logRegression(D,params=nsteps,trainMode='batch')

#i.procesar los datos por lotes (mini­batches)
errorDiff=0.01 # la condición de parada es la diferencia en el error de entrenamiento.
print 'Minibatch training:'
logRegression(D,params=[errorDiff,10],trainMode='minibatch')

#ii. utilizar como dataset Caltech101 (airplanes vs motorbikes) rescaleado a 28x28 pxl. 

dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/caltech/'
D=loadPlanesvsBikes(dataDir)



#iii. Agregar al modelo una capa de 100 neuronas ocultas con activación ReLU. 




# agregar pruebas con train/test
