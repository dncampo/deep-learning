import os, random, math, pandas
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from loadImages import loadImages
from normalizeImage import normalizeImage
from scaleRange import scaleRange
from getPatches import getPatches

plt.close('all')

dataDir='/home/leandro/workspace/Dropbox/ref/deeplearning_cifasis/data/caltech/'
# dataDir='/home/tc9/Escritorio/0_Copartida/Caltech101/'
imFiles=[]

for d in os.walk(dataDir):
    for f in d[2]:
        if ".directory" not in f:
            imFiles.append(d[0]+"/"+f)
            

# some dataset stats...
datasizes=np.zeros((len(imFiles),3))
for (k,f) in enumerate(imFiles):
    # tomar parametros de la base de datos
    im=misc.imread(f)
    datasizes[k,:]=(im.shape[0],im.shape[1],im.ndim)

fig = plt.figure()
fig.patch.set_facecolor('white')
imdata=pandas.DataFrame(datasizes)
imdata.boxplot()

maxSize=(128,128) 

print("Total: %d imágenes, %d en RGB\n" %(datasizes.shape[0],np.count_nonzero(datasizes[:,2]==3)))

#imFiles=imFiles[1:1000]

images=loadImages(imFiles,maxSize)

imej=[7,5]

f1,subplots1=plt.subplots(1,2)
f1.patch.set_facecolor('white')
for k  in [0,1]:
    subplots1[k].imshow(images[imej[k]])

# normalizo las imagenes originales
im1=misc.imread(imFiles[imej[0]])
im1n=normalizeImage(im1)
im2=misc.imread(imFiles[imej[1]])
im2n=normalizeImage(im2)

f2,subplots2=plt.subplots(2,2)
f2.patch.set_facecolor('white')
subplots2[0,0].imshow(im1)
subplots2[0,1].imshow(scaleRange(im1n))
subplots2[1,0].imshow(im2)
subplots2[1,1].imshow(scaleRange(im2n))
plt.show()

patches=getPatches(imFiles,N=10,S=16,norm=True)

f3,subplots3=plt.subplots(2,5)
f3.patch.set_facecolor('white')
for k1  in range(0,2):
    for k2  in range(0,5):
        subplots3[k1,k2].imshow(scaleRange(patches[imej[1],k1*5+k2]))
plt.show()

 
