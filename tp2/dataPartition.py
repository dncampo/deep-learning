import numpy as np
def dataPartition(data,ratio,rng):
    N=data[0].shape[0]
    ind=np.array(range(0,N))
    np.random.shuffle(ind)

    testind=ind[:round(N*ratio)]
    trainind=[n for n in range(0,N) if n not in testind]

    train=(data[0][trainind,:],data[1][trainind])
    test=(data[0][testind,:],data[1][testind])
    return train,test
