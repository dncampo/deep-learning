from read_image import *
from extract_window import *

def main():
    width = 450
    height = 450

    path = "/home/noname/Dropbox/testing_img"
    # path = "/home/noname/101_ObjectCategories/airplanes/"
    os.chdir(path)

    list_of_files = get_image_names(path)
    imgs = read_images(list_of_files)

    windows = extract_windows(list_of_files)




if __name__ == "__main__":
    main()