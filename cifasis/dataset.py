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


def read_image(path, max_width, max_height):
    """
    Returns a PIL image.
    The image gets resized.
    :param path: image path
    :param max_width: max image width
    :param max_height: max image height
    :return: three-dimensional numpy array
    """
    im = Image.open(path).resize((max_width, max_height), 1)  # filter=1 NEAREST
    return im


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
        if im_size[0] > 500 or im_size[1] > 500:
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
    print("Big images = {0}".format(n_big_images))


def image_to_tensor(img):
    """
    Returns a "tensorized" version of the image. Each pixel is represented by [value, channel, x_pos, y_pos].
    Channel values are R=0, G=1, B=2.
    :param img: PIL image
    :return: a vector with the tensorized version of the image
    """

    matrix_img = np.asarray(img)
    matrix_shape = matrix_img.shape

    vec = np.zeros(matrix_img.size * 4)
    idx = 0
    for row in range(matrix_shape[0]):
        for col in range(matrix_shape[1]):
            data = (matrix_img[row, col, 0], 0, row, col,
                    matrix_img[row, col, 1], 1, row, col,
                    matrix_img[row, col, 2], 2, row, col)
            vec[idx:idx + 12] = data
            idx += 12
    return vec



