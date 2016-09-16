import os
import numpy as np
from loadImages import loadImages

def loadPlanesvsBikes(dataDir):
    class=['Motorbikes/','airplanes/']
    S=28
    
    dataset=np.array()
    for c in [0,1]:
        imFiles=[]
        for d in os.walk(dataDir+class[c]):
            for f in d[2]:
                if ".directory" not in f:
                    imFiles.append(d[0]+"/"+f)
        N=len(imFiles)
        data=loadImages(imFiles,(S,S))
        features=np.reshape(data,[N,S*S*3])
        labels=c*np.ones([N,1])
        dataset1=np.concatenate((features,labels),axis=1)
        if not dataset:
            dataset=dataset1
        else:
            dataset=np.concatenate((dataset,dataset1),axis=0)
            
