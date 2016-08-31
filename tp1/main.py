import sys
sys.path.append("..")

from cifasis.dataset import *
from cifasis.imgfun import *


def main():
    width = 450
    height = 450
    
    path_images = parse_dataset_dir('/home/lpineda/101_ObjectCategories/')
    

    #dataset = read_dataset(path_images, width, height)
    
    test_img =  read_image(path_images[3958])
    test_img.show()
    
    cropped_img = extract_window(test_img)
    
    rr, gg, bb = test_img.split()
    
    quan_rr = quantize(normalize(np.array(bb)))
    quan_rr.show()
    

if __name__ == "__main__":
    main()
