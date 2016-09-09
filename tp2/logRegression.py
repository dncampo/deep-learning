import numpy as np
import theano
import theano.tensor as T
from accuracy import accuracy
rng = np.random

def logRegression(traindata,params,trainMode):

    x = T.dmatrix("x")
    y = T.dvector("y")
    feats=traindata[0].shape[1]
    
    # Inicializacion (shared para que mantengan valores entre updates)
    w = theano.shared(rng.randn(feats), name="w")
    b = theano.shared(0., name="b")

    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   
    prediction = p_1 > 0.5                    
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy

    cost = xent.mean() + 0.01 * (w ** 2).sum()
    gw, gb = T.grad(cost, [w, b])             
    # Compile
    train = theano.function(
        inputs=[x,y],
        outputs=[prediction, xent],
        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    if trainMode=='batch':
        training_steps=params
        for i in range(training_steps):
            pred_train, err = train(traindata[0], traindata[1])
            if i % 10 == 0:
                print 'Step %d: xent: %.03f, train acc: %.03f %%' %(i,err.mean(),accuracy(traindata[1],predict(traindata[0])))
 
    if trainMode=='minibatch':
        errordif=params[0]
        batchsize=params[1]
        e=1
        err_old=1
        while e>errordif:
            # tomar ejemplos al azar
            batchind=rng.randint(0,len(traindata[0]),batchsize)
            batchx=[traindata[0][i] for i in batchind]
            batchy=[traindata[1][i] for i in batchind]
            pred_train, err = train(batchx, batchy)
            e=np.abs((err-err_old)/err_old)

        
    # Test
    # pred, err = test(Dtest)
    #print('Resultado: %f %%' %accuracy(traindata[1],predict(traindata[0])))
