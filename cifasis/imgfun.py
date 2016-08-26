import random as rnd


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

    return img.crop([rnd_x, rnd_x + img_size[0],
                     rnd_y, rnd_y + img_size[1]])
