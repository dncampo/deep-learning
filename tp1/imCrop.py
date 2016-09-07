import math
import numpy as np

def imCrop(im,msize):

    if im.ndim<3: # copio canales
        im=np.array([im,im,im]).transpose()

    # Habría que fijar la dimensión principal en un sentido (si es 3:2 por ejemplo).
    # Para no perder información, lo lógico sería ampliar las imagenes mas chicas, centrarlas y rellenar con ceros para que cumpla el tamaño mas grande (sin perder relacion de aspecto).
    
        
    # Primero subsampleo para no perder tanta imagen
    subfactor=(math.floor(im.shape[0]/msize[0]),math.floor(im.shape[1]/msize[1]))
    if subfactor[0]>0:
        im=im[::subfactor[0],:,:]
    if subfactor[1]>0:
        im=im[:,::subfactor[1],:]

    print(subfactor)
        
    dx=math.floor(im.shape[0]/2-msize[0]/2)
    dy=math.floor(im.shape[1]/2-msize[1]/2)

    # 0 ganas de pensar, esto se debe poder hacer por diensión
    if dx!=0:
        im=im[dx:-dx,:,:]
    if dy!=0:
        im=im[:,dy:-dy,:]
        
    if im.shape[0]>msize[0]:
        im=im[1:,:,:]
    if im.shape[1]>msize[1]:
        im=im[:,1:,:]
        
        
    return im
