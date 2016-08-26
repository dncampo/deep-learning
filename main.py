from cifasis.dataset import *
from cifasis.imgfun import *


def main():
    width = 200
    height = 300
    path_images = parse_dataset_dir('/home/lpineda/101_ObjectCategories/')
    n_images = len(path_images)

    rnd_img_path = path_images[int(rnd.random()*n_images)]
    rnd_img = read_image(rnd_img_path, width, height)
    rnd_img_tensor = image_to_tensor(rnd_img)
    print(rnd_img_tensor[120:160])

    scan_dataset(path_images)

    rnd_img.show()
    sub_img = extract_window(rnd_img)
    sub_img.show()


if __name__ == "__main__":
    main()
