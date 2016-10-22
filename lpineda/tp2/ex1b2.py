from sklearn.cross_validation import train_test_split

import sys
sys.path.append("..")
from cifasis.dataset import *

import matplotlib.pyplot as plt

import numpy
import theano
import theano.tensor as T
rng = numpy.random

#N = 400                                   # training sample size
feats = 784*3                               # number of input variables

# generate a dataset: D = (input_values, target_class)
#D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

print("Parsing dataset")
airplanes_path = '../../101_ObjectCategories/airplanes'
motorbikes_path = '../../101_ObjectCategories/Motorbikes'
D1 = read_dataset(parse_dataset_dir(airplanes_path), 28, 28, transformation='reshape')
D1 = D1/np.max(D1)
n_airplanes = len(D1[:,1])
D2 = read_dataset(parse_dataset_dir(motorbikes_path), 28, 28, transformation='reshape')
D2 = D2/np.max(D2)
n_motorbikes = len(D2[:,1])
D = (np.vstack((D1, D2)), np.hstack((np.zeros(n_airplanes), np.ones(n_motorbikes))))

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

#print("Initial model:")
#print(w.get_value())
#print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()  # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)
lrn_rate = 0.01

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - lrn_rate * gw), (b, b - lrn_rate * gb)))
          
predict = theano.function(inputs=[x], outputs=prediction)

# Train
data_train, data_test, labels_train, labels_test = train_test_split(D[0], D[1], test_size=0.3)

train_error = []
test_error = []
epoch = 0
max_epoch = 500

queue = list(range(50))

print("Training started")
while epoch < max_epoch and np.var(queue)>10e-10:
    epoch += 1
    #Training phase
    print("Epoch " + str(epoch) + " of " + str(max_epoch))
    for batch in get_mini_batches((data_train, labels_train), size=64):
        pred, err = train(batch[0], batch[1])
#    pred, err = train(data_train, labels_train)
    pred_train = predict(data_train)
    train_error.append(sum(pred_train != labels_train)/float(len(labels_train)))
    
    #Test phase
    pred_test = predict(data_test)
    test_error.append(sum(pred_test != labels_test)/float(len(labels_test)))
    queue.pop()
    queue.insert(0,test_error[-1])
    
if epoch < max_epoch:
    print("Ended by test criteria")
else:
    print("Maximum number of epochs has been reached")

plt.plot(range(len(train_error)),train_error)
plt.plot(range(len(test_error)),test_error)
plt.legend(['Train error','Test error'])
plt.xlabel('Epoch')
plt.ylabel('% error')
plt.savefig('ex1b2.png')

#print("Final model:")
#print(w.get_value())
#print(b.get_value())
#print("target values for D:")
#print(D[1])
#print("prediction on D:")
#print(predict(D[0]))
