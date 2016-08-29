import sys
sys.path.append("..")

from cifasis.dataset import *
from cifasis.imgfun import *


def main():
    width = 450
    height = 450
    path_images = parse_dataset_dir('/home/lpineda/101_ObjectCategories/')

    dataset = read_dataset(path_images, width, height)
    
    print(dataset.shape)


if __name__ == "__main__":
    main()
