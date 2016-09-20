from cifasis.dataset import *
from sklearn.cross_validation import train_test_split

import sys
sys.path.append("..")

import matplotlib.pyplot as plt

import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
max_training_steps = 1000


# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)


# Train

data_train, data_test, labels_train, labels_test = train_test_split(D[0], D[1], test_size=0.2)

train_error = []
test_error = []
step = 0

print("Training started")
while step < max_training_steps:
    step += 1
    #Training phase
    sys.stdout.write(str(step) + " ")
    for batch in get_mini_batches((data_train, labels_train)):
        pred, err = train(batch[0], batch[1])
    pred_train = predict(data_train)
    train_error.append(sum(pred_train != labels_train)/float(len(labels_train)))
    
    #Test phase
    pred_test = predict(data_test)
    test_error.append(sum(pred_test != labels_test)/float(len(labels_test)))

plt.plot(range(len(train_error)),train_error)
plt.plot(range(len(test_error)),test_error)

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
