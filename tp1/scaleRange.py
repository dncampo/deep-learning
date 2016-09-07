def scaleRange(im):
    '''Escalar imagene en el rango [0-1] para graficar'''
    im2=im.copy()
    im2=(im2-im2.min())/(im2.max()-im2.min())
    return im2
    
