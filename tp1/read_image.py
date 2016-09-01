import os
import numpy as np
from PIL import Image


image_height = 400
image_width = 400

def read_images(filenames):
    """
    Returns a numpy matrix with all images read. Each row of the matrix
    represents a single image. An image is stored as a continous set of
    pixels with its RGB values.
    :param filenames: absolute path of all images to read
    :param max_width: max image width
    :param max_height: max image height
    :return: a numpy matrix
    """
    print len(filenames)
    max_w = 0
    max_h = 0
    for f in filenames:
        im = Image.open(filenames[f])
        img = np.array(im)
        img_s = img.shape
        if img_s[0] > max_w:
            max_w = img_s[0]
        if img_s[1] > max_h:
            max_h = img_s[1]

    print max_w
    print max_h
    global image_width
    image_width = max_w
    global image_height
    image_height = max_h

    img_flatten_size = max_w*max_h*3
    images = np.zeros((len(filenames), img_flatten_size), dtype=np.uint8)
    i = 0

    for f in filenames:
        im = Image.open(filenames[f])
        if im.mode != 'RGB':
            im.convert('RGB')
        img = np.array(im)
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        canvas[0:img.shape[0], 0:img.shape[1]] = img

        img_f = canvas.flatten('C')

        images[i, 0:img_f.size] = img_f
        i = i + 1

    return images


def reconstruct_image(a_row):
    """
    Returns a PIL Image from a vector representing an image
    :param a_row: a vector row representing an image
    :return: a PIL Image
    """
    r = a_row[0::3].reshape(image_width, image_height)
    g = a_row[1::3].reshape(image_width, image_height)
    b = a_row[2::3].reshape(image_width, image_height)

    rgb = np.zeros((r.shape[0], r.shape[1], 3), 'uint8')
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b

    im = Image.fromarray(rgb)
    return im


def get_image_names (from_path):
    """
    Returns a list of strings representing the absolute path of each
    jpg image.
    :param from_path: starting path
    :return: a list of strings
    """
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(from_path):
        if dirnames is 'no' or dirpath is 'NO':
            continue
        for filename in filenames:
            if filename.endswith('.jpg'):
                list_of_files[filename] = os.sep.join([dirpath, filename])
                print os.sep.join([dirpath, filename])
    return list_of_files


