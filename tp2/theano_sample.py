#!/usr/bin/python
# -*- coding: utf-8 -*-
import theano
import numpy as np
from logRegression import logRegression
from loadPlanesvsBikes import loadPlanesvsBikes
rng = np.random

ej='bii' # bi,bii,c

# TP2.1.a: 
# Revise​este tutorial de theano​ y resuelva el ejercicio propuesto. 
if ej=='a':
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
if ej=='bi':
    nsteps = 300
    print('Batch training:')
    logRegression(D,params=nsteps,trainMode='batch')

#i.procesar los datos por lotes (mini­batches)
if ej=='bi':
    errorDiff=0.01 # la condición de parada es la diferencia en el error de entrenamiento.
    print('Minibatch training:')
    logRegression(D,params=[errorDiff,10],trainMode='minibatch')

#ii. utilizar como dataset Caltech101 (airplanes vs motorbikes) rescaleado a 28x28 pxl. 
errorDiff=0.01
dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/caltech/'
#dataDir='/home/tc9/Descargas/101_ObjectCategories/'
D=loadPlanesvsBikes(dataDir)

if ej=='bii':
    print('Airplanes vs Motorbikes:')
    logRegression(D,params=[errorDiff,10],trainMode='minibatch')



#iii. Agregar al modelo una capa de 100 neuronas ocultas con activación ReLU.
# TODO: mostrar la estructura de la red, verificar que los pesos se actualizan.
if ej=='biii':
    print('Airplanes vs Motorbikes + hidden layer:')
    logRegression(D,params=[errorDiff,10],trainMode='minibatch',nh=100)




# agregar pruebas con train/test
