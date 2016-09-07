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
   
    if trainMode=='minibatch':
        errordif=params
        e=1
        while e>errordif:
            # tomar N ejemplos al azar
            pred_train, err = train(traindata[0], traindata[1])

        
    # Test
    # pred, err = test(Dtest)
    print('Resultado: %f %%' %accuracy(traindata[1],predict(traindata[0])))
