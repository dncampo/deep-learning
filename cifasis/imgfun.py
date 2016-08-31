import random as rnd
import numpy as np
from PIL import Image

def extract_window(img, window_size=None):
    """
    Returns a random subimage from the image given
    :param img: the image
    :param window_size: subimage window size
    :return: a subimage
    """
    if window_size is None:  # check http://docs.python-guide.org/en/latest/writing/gotchas/
        window_size = [16, 16]

    img_size = img.size
    
    # if the image is smaller than window_size, a subimage cannot be extracted
    if img_size[0] < window_size[0] or img_size[1] < window_size[1]:
        raise Exception("Can't extract subimage")

    # generate a random top-left (x,y) position
    rnd_x = int(rnd.random()*(img_size[0]-window_size[0]))
    rnd_y = int(rnd.random()*(img_size[1]-window_size[1]))

    return img.crop((rnd_x, rnd_y, rnd_x + window_size[0], rnd_y + window_size[1]))
    
def normalize(np_img):
    """
    Returns np_img with mean equal to zero and std dev equal to one
    :param np_img: numpy matrix
    """

    mean = np.mean(np_img)
    std = np.std(np_img)
    
    np_img = (np_img - mean) / std
    
    return np_img

def quantize(np_img):
    """
    Returns a quantized version of np_img in the range [0, 255].
    :param np img: numpy matrix
    """
    
    min_img = np.min(np_img)
    max_img = np.max(np_img)
    
    range_img = max_img - min_img
    
    f = np.vectorize(lambda x: np.int8(255*(x-min_img)/range_img))
    
    quantized_img = f(np_img)
    img =  Image.fromarray(quantized_img,"L")

    return img
    
    
    