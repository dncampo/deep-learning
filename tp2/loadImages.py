import skimage.transform as sit
import numpy as np
from scipy import misc
import gc

def loadImages(files,maxsize):
    
    images=np.zeros([len(files),maxsize[0],maxsize[1],3],dtype=np.byte) 

    for (k,f) in enumerate(files):
        im=misc.imread(f)
        #print(im.shape)
        if im.ndim==2: # escala de grises
            im=np.swapaxes(np.array([im,im,im]),0,2)
            im=np.swapaxes(im,0,1)
        if im.shape[2]==4: # png, eliminar capa de transparencia
            im=im[:,:,0:3]
            
        im=sit.rescale(im,min(maxsize[0]/im.shape[0],maxsize[1]/im.shape[1]))

        # zero pad 
        if im.shape[1]<maxsize[1]:
            cols=maxsize[1]-im.shape[1]
            im=np.concatenate((im,np.zeros([im.shape[0],cols,3])),1)
        if im.shape[0]<maxsize[0]:
            rows=maxsize[0]-im.shape[0]
            im=np.concatenate((im,np.zeros([rows,im.shape[1],3])),0)

            
        #print(im.shape)
        images[k]=im
        gc.collect()
    return images

