#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from accuracy import accuracy


def logRegression(traindata,testdata,params,trainMode,rng,nh=0,hfun='sig'):

  
    x = T.dmatrix("x")
    y = T.dvector("y")
    feats=traindata[0].shape[1]

    if nh>0:
        wh = theano.shared(rng.randn(feats,nh), name="wh")
        bh = theano.shared(rng.randn(nh)*0., name="bh")
        wo = theano.shared(rng.randn(nh), name="wo")
        bo = theano.shared(0., name="bo")
        if hfun=='sig':
            h = 1 / (1 + T.exp(-T.dot(x, wh) - bh))   
        if hfun=='relu':
            #h=relu(T.dot(x, wh)+bh)
            h=theano.tensor.nnet.relu(T.dot(x, wh)+bh) 
        # final output
        p_1 = 1 / (1 + T.exp(-T.dot(h, wo) - bo))   
    else:
        # Inicializacion (shared para que mantengan valores entre updates)
        w = theano.shared(rng.randn(feats), name="w")
        b = theano.shared(0., name="b")
        p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   
    
    
    prediction = p_1 > 0.5                    
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy


    if nh>0:
        cost = xent.mean() + 0.01 * (wo ** 2).sum() + 0.01 * (wh ** 2).sum()
        gwo, gbo, gwh, gbh = T.grad(cost, [wo, bo, wh, bh] )

        updates=((wo, wo - 0.1 * gwo), (bo, bo - 0.1 * gbo),
                 (wh, wh - 0.1 * gwh), (bh, bh - 0.1 * gbh))
    else:
        cost = xent.mean() + 0.01 * (w ** 2).sum()
        gw, gb = T.grad(cost, [w, b])
        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
    

    # Compile
    train = theano.function(
        inputs=[x,y],
        outputs=[prediction, xent],
        updates=updates)
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    if trainMode=='batch':
        training_steps=params
        for step in range(training_steps):
            pred_train, err = train(traindata[0], traindata[1])
            # if step %50 == 0:
            #      print('Step %d: xent: %.03f, train acc: %.03f %%' %(step,err.mean(),accuracy(traindata[1],predict(traindata[0]))))
        trainacc=accuracy(traindata[1],predict(traindata[0]))
        #print('Step %d: xent: %.03f, train acc: %.03f %%' %(step,err.mean(),accuracy(traindata[1],predict(traindata[0]))))
 
    if trainMode=='minibatch':
        errordif=params[0]
        batchsize=params[1]
        maxsteps=params[2]
        e=1
        err_old=1
        step=0
        while e>errordif and step<maxsteps:
            # tomar ejemplos al azar
            batchind=rng.randint(0,len(traindata[0]),batchsize)
            batchx=[traindata[0][i] for i in batchind]
            batchy=[traindata[1][i] for i in batchind]
            if 'err' in locals():
                err_old=err.mean()
            pred_train, err = train(batchx, batchy)
            # Tomo la xent como condición de stop
            e=np.abs((err.mean()-err_old)/err_old)
            # if step % 50 == 0:
            #print('Step %d: xent: %.03f, ediff: %.03f, train acc: %.03f %%' %(step,err.mean(),e,accuracy(batchy,predict(batchx)))) # 
            step+=1
        trainacc=accuracy(batchy,predict(batchx))
        #print('Step %d: xent: %.03f, ediff: %.03f, train acc: %.03f %%' %(step,err.mean(),e,accuracy(batchy,predict(batchx))))
        
    # Test
    pred = predict(testdata[0])
    testacc=accuracy(testdata[1],pred)
    # print('Test:  acc: %.03f' %accuracy(testdata[1],pred))
    # print('Balance Test:  %d clase0 - %d clase1' %(sum([1 for s in testdata[1] if s==0]), sum([1 for s in testdata[1] if s==1])))
    # print('Balance Train:  %d clase0 - %d clase1' %(sum([1 for s in traindata[1] if s==0]), sum([1 for s in traindata[1] if s==1])))
    return trainacc,testacc
