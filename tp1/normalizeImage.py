import numpy as np

def normalizeImage(im):
    '''zscore normalization'''
    imn=im.astype(float)
    for c in [0,1,2]:
        imn[:,:,c]=im[:,:,c]-np.mean(im[:,:,c])
        imn[:,:,c]=imn[:,:,c]/np.std(im[:,:,c])
        
    return imn
