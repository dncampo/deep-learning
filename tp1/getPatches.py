import numpy as np
from scipy import misc
import random
from normalizeImage import normalizeImage
def getPatches(imageFiles,N,S,norm,fixed=False):
    '''Obtener N parches de SxS pìxeles en posiciones aleatorias de las imagenes'''
    patches=np.zeros([len(imageFiles),N,S,S,3])
    if fixed:
        random.seed(15)
    
    for i in range(0,len(imageFiles)):
        im=misc.imread(imageFiles[i])
        if im.ndim==2: # escala de grises
            im=np.swapaxes(np.array([im,im,im]),0,2)
            im=np.swapaxes(im,0,1)
        if im.shape[2]==4: # png
            im=im[:,:,0:3]
        for n in range(0,N):
            pos=(random.randint(S/2,im.shape[0]-S/2),random.randint(S/2,im.shape[1]-S/2))
            patches[i,n]=im[pos[0]-S/2:pos[0]+S/2,pos[1]-S/2:pos[1]+S/2,:]
            if norm:
                patches[i,n]=normalizeImage(patches[i,n])

    return patches
