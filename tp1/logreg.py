from numpy import random as rng
import theano 
import theano.tensor as T
feats=784

x=T.dmatrix('x')
y=T.dvector('y')
w=theano.shared(rng.randn(feats),name='w')
b=theano.shared(0.,name='b')

p1=1/(1+T.exp(-T.dot(x,w)-b))
prediction=p1>0.5

xent=-y*T.log(p1)-(1-y)*T.log(1-p1) #cross-entropy / 1loglikelihood?
cost=xent.mean()+0.01*(w**2).sum()
gw,gb=T.grad(cost,[w,b])
train=theano.function(inputs=[x,y],outputs=[prediction,xent,cost],updates=((w,w-0.1*gw), (b,b-0.1*gb)))

predict=theano.function(inputs=[x],outputs=prediction)
