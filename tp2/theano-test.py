# pruebas
import numpy 
import theano.tensor as T
from theano import function

x=T.matrix('x',dtype='float32')
y=T.matrix('y',dtype='float32')
z=x+y
f=function([x,y],z)

a=[[1,2],[3,4]]
b=[[10,20],[30,40]]
c=[[10,20,30],[30,40,50]]

print f(a,b)
#print f.name

from theano import pp
print pp(z)

import theano

def relu(x):
    return T.switch(x>0.0,x,0.0)

x=T.vector('x')
W=T.matrix('W')
b=T.vector('b')
z=T.dot(x,W)+b
y=relu(z)

layer=theano.function([x,W,b],y)
print layer([3,1],[[1,0,0],[0,-4,0]],[1,-2,3])

from theano import shared

state=shared(0)
inc=T.iscalar('inc')
accumulator=function([inc],state,updates=[(state,state+inc)])
state.set_value(10)

print state.get_value()
accumulator(2)
print state.get_value()
print state.get_value()

from theano.tensor.shared_randomstreams import RandomStreams

srng= RandomStreams(seed=234)
rv_u=srng.uniform((2,))
rv_n=srng.normal((2,))

f=function([],rv_u)
g=function([],[rv_u,rv_n],no_default_updates=True)
d=function([],[2*rv_u,rv_u+rv_u])

print f()
print g()
print f()
print f()
print d()

x=T.dscalar('x') #dscalar es 64bits
y=x**2
gy=T.grad(y,x)
print theano.pp(gy)

g=theano.function([x],gy)
print g(4)
print theano.pp(g.maker.fgraph.outputs[0])

# Reducciones
x=T.tensor3()
total=x.sum()
marginals=x.sum(axis=(0,2))
mx=x.max(axis=1)

# dimshuffle
y=x.dimshuffle((2,1,0))
print y

a=T.matrix()
b=a.T
c=a.dimshuffle((1, 0)) # Same as b
d=a.dimshuffle((0, 1, 'x'))
e=a + d
f=function([a],[b,c,d,e])

print f([[1,2],[3,4]]) 
