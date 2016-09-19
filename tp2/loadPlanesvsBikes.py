import os
import numpy as np
from loadImages import loadImages

def loadPlanesvsBikes(dataDir):
    classes=['Motorbikes/','airplanes/']
    S=28
    
    for c in [0,1]:
        imFiles=[]
        for d in os.walk(dataDir+classes[c]):
            for f in d[2]:
                if ".directory" not in f:
                    imFiles.append(d[0]+"/"+f)
        N=len(imFiles)
        data=loadImages(imFiles,(S,S))
        print(data.shape)
        features=np.reshape(data,[N,S*S*3])
        labels=c*np.ones([N,1])
        dataset1=np.concatenate((features,labels),axis=1)
        try:
            dataset=np.concatenate((dataset,dataset1),axis=0)
        except:
            dataset=dataset1
    print("dataset size:")
    print(dataset.shape)
    return dataset
