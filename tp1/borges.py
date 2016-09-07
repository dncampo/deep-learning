import os, random, math, pandas
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from loadImages import loadImages
from scaleRange import scaleRange
from getPatches import getPatches
# borrar: cosas para sacar cuando este listo para presentar

dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/deep-learning-course/test/words/'
imFiles=[]

for d in os.walk(dataDir):
    for f in d[2]:
        if ".directory" not in f:
            imFiles.append(d[0]+"/"+f)

# Para evaluar las características del dataset:
# datasizes=np.zeros((len(imFiles),3))
# for (k,f) in enumerate(imFiles):
#     im=misc.imread(f)
#     datasizes[k,:]=(im.shape[0],im.shape[1],im.ndim)    
# imdata=pandas.DataFrame(datasizes)
# fig = plt.figure()
# fig.patch.set_facecolor('white')
# imdata.boxplot()
# plt.show()

maxSize=(128,128)

images=loadImages(imFiles,maxSize)

patches=getPatches(imFiles,N=10,S=16,norm=True,fixed=True)
f1,subplots1=plt.subplots(2,5)
f1.patch.set_facecolor('white')
for k1  in range(0,2):
    for k2  in range(0,5):
        im2plot=scaleRange(patches[0,k1*5+k2])
        subplots1[k1,k2].imshow(im2plot)

patches=getPatches(imFiles,N=10,S=16,norm=False,fixed=True)
f2,subplots2=plt.subplots(2,5)
f2.patch.set_facecolor('white')
for k1  in range(0,2):
    for k2  in range(0,5):
        im2plot=scaleRange(patches[0,k1*5+k2])
        subplots2[k1,k2].imshow(im2plot)
plt.show()
