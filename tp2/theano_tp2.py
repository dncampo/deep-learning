#!/usr/bin/python
# -*- coding: utf-8 -*-
import theano
import sys
import numpy as np
from logRegression import logRegression
from loadPlanesvsBikes import loadPlanesvsBikes
np.random.seed(700)  # for reproducibility
rng = np.random

dataset='mvsp' #rand,mvsp
ej='bii-biii' # a,bi,bii-biii

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
    print(g([0, 1, 2],[0, 2, 3]))

if dataset=='rand':
    N = 400               
    feats = 784           
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
if dataset=='mvsp':
    dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/caltech/'
    D=loadPlanesvsBikes(dataDir)
    
    
# TP2.1.b:
# Modifique​este ejemplo de regresióxn logística​ para: 
# generate a dataset: D = (input_values, target_class)

if ej=='bi':
    nsteps = 50
    print('Batch training:')
    logRegression(D,params=nsteps,trainMode='batch')

#i.procesar los datos por lotes (mini­batches)
if ej=='bi':
    errorDiff=0.01 # la condición de parada es la diferencia en el error de entrenamiento.
    print('Minibatch training:')
    logRegression(D,params=[errorDiff,10],trainMode='minibatch')

#ii. utilizar como dataset Caltech101 (airplanes vs motorbikes) rescaleado a 28x28 pxl. 
#iii. Agregar al modelo una capa de 100 neuronas ocultas con activación ReLU.
if ej=='bii-biii':
    print('Airplanes vs Motorbikes:')
    errorDiff=0.001
    logRegression(D,params=[errorDiff,10],trainMode='minibatch')
    print('Airplanes vs Motorbikes + hidden layer sigmoidea:')
    logRegression(D,params=[errorDiff,10],trainMode='minibatch',nh=100,hfun='sig')
    print('Airplanes vs Motorbikes + hidden layer relu:')
    logRegression(D,params=[errorDiff,10],trainMode='minibatch',nh=100,hfun='relu')


