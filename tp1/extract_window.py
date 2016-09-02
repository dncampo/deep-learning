from PIL import Image
import random as r
import numpy as np


def normalize_img (an_img):
    img_np = np.array(an_img)
    num_pixels = img_np[..., 0].size
    img_float = img_np.astype(float)
    an_img.show()
    mean = np.zeros(3)
    var = mean

    im = Image.fromarray(img_np[..., 0])
    im.show()

    for i in range(0, 3):
        mean[i] = np.sum(img_np[..., i]) / num_pixels
        var[i] = np.sum((img_float - mean[i])**2) / num_pixels
        img_float[..., i] = (img_float[..., i] - mean[i]) / var[i]
        im = (img_float[...,i])
        im = Image.fromarray(img_float[...,i].astype(np.uint8))
        im.show()


    im = (img_float*128)+128
    im = Image.fromarray(img_float.astype(np.uint8))
    #im.show()



def extract_windows(filenames, x_win=500, y_win=500, num_win =1 ):
    for f in filenames:
        im = Image.open(filenames[f])


        if x_win > im.size[0]:
            x_win = im.size[0]
        if y_win > im.size[1]:
            y_win = im.size[1]

        for i in range(num_win):
            win = extract_window(im, x_win, y_win)
            #win.show()




def extract_window(im, x_win, y_win):
    random_x = int(r.random() * (im.size[0] - x_win))
    random_y = int(r.random() * (im.size[1] - y_win))
    win = im.crop((random_x, random_y, random_x + x_win, random_y + y_win))

    normalize_img(win)
    return win