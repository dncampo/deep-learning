#!/usr/bin/python
# -*- coding: utf-8 -*-
import theano
import sys
import numpy as np
from dataPartition import dataPartition
from logRegression import logRegression
from loadPlanesvsBikes import loadPlanesvsBikes
import time
np.random.seed(750)
rng = np.random

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


    
# TP2.1.b:
# Modifique​este ejemplo de regresióxn logística​ para: 
# generate a dataset: D = (input_values, target_class)

if ej=='bi':
    N = 400               
    feats = 784           
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    # 20% para test
    traindata,testdata=dataPartition(D,.20)


    nsteps = 50
    print('Batch training:')
    start=time.time()
    logRegression(traindata,testdata,params=nsteps,trainMode='batch')
    print('Elapsed time: %.02f s\n' %(time.time()-start))

    #i.procesar los datos por lotes (mini­batches)
    errorDiff=0.01 # la condición de parada es la diferencia en el error de entrenamiento.
    print('Minibatch training:')
    start=time.time()
    logRegression(traindata,testdata,params=[errorDiff,10],trainMode='minibatch')
    print('Elapsed time: %.02f s\n' %(time.time()-start))

#ii. utilizar como dataset Caltech101 (airplanes vs motorbikes) rescaleado a 28x28 pxl. 
#iii. Agregar al modelo una capa de 100 neuronas ocultas con activación ReLU.
if ej=='bii-biii':

    dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/caltech/'
    D=loadPlanesvsBikes(dataDir)

    seeds=[750,2,123,489,1000,656,22,123,159,953]
    acctest=np.zeros((4,len(seeds)))
    acctrain=np.zeros((4,len(seeds)))
    etime=np.zeros((4,len(seeds)))
    
    # algunas repeticiones
    for (k,seed) in enumerate(seeds):
        print('run %d' %k)
        np.random.seed(seed)
        rng = np.random

        # 20% para test
        traindata,testdata=dataPartition(D,.20,rng)
        nsteps = 500
        errorDiff=0.001
    
        for e in range(0,acctest.shape[0]):

            print('run %d: %d' %(k,e))
        
            start=time.time()
            if e==0:
                atrain,atest=logRegression(traindata,testdata,params=nsteps,trainMode='batch',rng=rng)
            if e==1:
                atrain,atest=logRegression(traindata,testdata,params=[errorDiff,10,nsteps],trainMode='minibatch',rng=rng  )
            if e==2:
                atrain,atest=logRegression(traindata,testdata,params=[errorDiff,10,nsteps],trainMode='minibatch',rng=rng,nh=100,hfun='sig')
            if e==3:
                atrain,atest=logRegression(traindata,testdata,params=[errorDiff,10,nsteps],trainMode='minibatch',rng=rng,nh=100,hfun='relu')
           
            acctrain[e,k]=atrain
            acctest[e,k]=atest
            etime[e,k]=time.time()-start
        
    for e in range(0,acctest.shape[0]):
        print('%d' %e)
        print('%.02f (SD: %.02f)'%(acctrain[e,:].mean(),acctrain[e,:].std()))
        print('%.02f (SD: %.02f)'%(acctest[e,:].mean(),acctest[e,:].std()))
        print('%.02f (SD: %.02f)'%(etime[e,:].mean(),etime[e,:].std()))
        
   
  
