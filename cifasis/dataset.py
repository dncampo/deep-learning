import numpy as np
from PIL import Image
import os


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


def reshape_image(path, max_width, max_height):
    """
    Returns a PIL image with black borders. If the image size exceed the parameters given, an exception is raised.
    :param path: image path
    :param max_width: max image width
    :param max_height: max image height
    :return: PIL image
    """
    
    im = Image.open(path)
    im_width = im.size[0]
    im_height = im.size[1]
    if im_width > max_width or im_height > max_height:
        raise Exception("Image size cannot exceed width or height parameters" + path)
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
    This function returns None. Just print relevant information about the dataset
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
    dataset = np.zeros((n_images, vector_len)) 
    
    for i in range(n_images):
        clamped_img = reshape_image(path_list[i], max_width, max_height)
        dataset[i,:] = np.reshape(clamped_img, (1,vector_len))
        
    return dataset
        



