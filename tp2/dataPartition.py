import numpy as np
rng = np.random
def dataPartition(data,ratio):
    N=data[0].shape[0]
    testind=rng.randint(0,N,round(N*ratio))
    trainind=[n for n in range(0,N) if n not in testind]
    train=(data[0][trainind,:],data[1][trainind])
    test=(data[0][testind,:],data[1][testind])
    return train,test
