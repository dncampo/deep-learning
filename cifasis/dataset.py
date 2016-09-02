import numpy as np
from PIL import Image
import os
import re
import operator


def parse_dataset_dir(path):
    """
    Returns a list of strings, with the absolute path of each image.
        This implementation is recursive and supports subdirectories.
    :param path: dataset directory path
    :return: a list of strings
    """

    return parse_dataset_dir_wrapper(path, [])


def parse_dataset_dir_wrapper(path, accum):
    """
    parseDatasetDir wrapper dataset
    :param path: dataset directory path
    :param accum: accumulator used to passing information between dataset calls
    :return: a list of strings
    """
    if os.path.isfile(path):
        accum += [path]
        return accum
    else:
        for p in os.listdir(path):
            parse_dataset_dir_wrapper(path + '/' + p, accum)
    return accum

def read_image(path):
    im = Image.open(path)
    if im.mode != 'RGB':
        return im.convert('RGB')
    return im

    
def reshape_image(im, max_width, max_height):
    """
    Returns a PIL image with black borders. If the image size exceed the parameters given, an exception is raised.
    :param path: image path
    :param max_width: max image width
    :param max_height: max image height
    :return: PIL image
    """
    
    im_width = im.size[0]
    im_height = im.size[1]
    if max_width % 2 or max_height % 2:
        raise Exception("Why would anyone want to use odd sizes...?")
        
    if im_width % 2 != 0:
        im_width -= 1
    if im_height % 2 != 0:
        im_height -= 1
    
    width_left =  max_width - im_width
    height_left = max_height - im_height
    
    new_im = Image.new('RGB',(max_width,max_height))
    new_im.paste(im,(int(width_left/2), int(height_left/2)))
    return new_im


def scan_dataset(path):
    """
    This function returns None. Just prints relevant information about the dataset
    :param path: a list of strings
    :return: None
    """

    n_images = len(path)
    mean_x = 0
    mean_y = 0
    max_width = 0
    max_height = 0
    n_big_images = 0  # an image is considered big if it has more than 500 pixels in any direction

    for p in path:
        im = Image.open(p)
        im_size = im.size
        if im_size[0] > 450 or im_size[1] > 450:
            n_big_images += 1
        else:
            mean_x += im_size[0]
            mean_y += im_size[1]
            if im_size[0] > max_width:
                max_width = im_size[0]
            if im_size[1] > max_height:
                max_height = im_size[1]

    mean_x /= n_images
    mean_y /= n_images

    print("Max width {0} max height {1}. Mean x {2} mean y {3}".format(max_width, max_height, mean_x, mean_y))
    print("There are {0} images. Big images = {1}".format(n_images, n_big_images))


def read_dataset(path_list, max_width, max_height):
    """
    Returns a numpy matrix with the normalized dataset
    Channel values are R=0, G=1, B=2.
    :param path_list: a list of strings
    :param max_width: max image width
    :param max_height: max image height
    :return: a numpy matrix
    """

    n_images = len(path_list)
    vector_len = max_width*max_height*3 # RGB 3 channels
    print("Each image has {0} values.".format(max_width*max_height*3))
    print("Each pixel is 64 bits so each image is {0} bits.".format(max_width*max_height*3*64))
    print("So each image has {0:.2f} mb.".format(max_width*max_height*3.0*64/8/2**20))
    print("The hole dataset is {0:.2f} gb.".format((max_width*max_height*3.0*64*len(path_list)/8/(2**30))))
    try:    
        dataset = np.zeros((n_images, vector_len)) 
    except MemoryError:
        print("ERROR: The array does not fit in memory")
        raise

    for i in range(n_images):
        img = read_image(path_list[i])
        if img.size[0] <= max_width and img.size[1] <= max_height:
            clamped_img = reshape_image(img, max_width, max_height)
            dataset[i,:] = np.reshape(clamped_img, (1,vector_len))
    
    print("Dataset shape: {0}".format(dataset.shape))
    return dataset
    
def get_nietszche_word_count(path):
    """
    Returns a sorted list of tuples (string, int).
    :param path: text file path
    """
    sentences = sentences_nietzsche_dataset(path)
    words = map(lambda x: x.split(), sentences)
    words = map(lambda x: (x, 1), [item for sublist in words for item in sublist])
    
    d = dict()
    
    for t in words:
        if t[0] in d:
            d[t[0]] += 1
        else:
            d[t[0]] = 1
    
    return sorted(d.items(), key=operator.itemgetter(1), reverse=True)

        
def sentences_nietzsche_dataset(path):
    """
    Returns a vector of strings. Each string is a sentence.
    :param path: text file path
    """
    
    f = open(path,'r')
    s = f.read()
    sentences = s.replace("\n"," ").replace("...",".").split(".")
    sentences = map(lambda x: re.sub( '\s+', ' ', x).strip(), sentences)
    sentences = map(lambda x: x.lower(), sentences)
    sentences = map(lambda x: re.sub('([^0-9a-zA-Z\s]+)', '', x), sentences)

    return sentences  
    
def map_sentence(sentence, word_count):
    """
    Returns a vector of integers, one for each word.
    :param sentence: a string
    :param word_count: a dict string to int    
    """
    i=0
    freq = np.zeros(len(sentence.split()), dtype=int)
    
    for w in sentence.split():
        freq[i] = word_count.get(w,0)
        i+=1
    
    return freq
    
