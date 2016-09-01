from PIL import Image
import random as r


def extract_windows(filenames, x_win=30, y_win=30, num_win = 15):
    for f in filenames:
        im = Image.open(filenames[f])


        if x_win > im.size[0]:
            x_win = im.size[0]
        if y_win > im.size[1]:
            y_win = im.size[1]

        for i in range(num_win):
            win = extract_window(im, x_win, y_win)
            win.show()




def extract_window(im, x_win, y_win):
    random_x = int(r.random() * (im.size[0] - x_win))
    random_y = int(r.random() * (im.size[1] - y_win))
    win = im.crop((random_x, random_y, random_x + x_win, random_y + y_win))

    return win