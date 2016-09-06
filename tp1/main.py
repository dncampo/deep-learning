import sys
sys.path.append("..")

from cifasis.dataset import *
from cifasis.imgfun import *

from matplotlib.mlab import PCA

def main():
    width = 500
    height = 500
    
#    ## exercise 1
#    path_images = parse_dataset_dir('/home/lpineda/101_ObjectCategories/laptop')
#    #dataset = read_dataset(path_images, width, height)
#
#
#    ## exercise 2
#    test_img =  read_image(path_images[12])
#    test_img.save('image.jpg')
#    plot_hist(test_img.histogram(),"image_histogram")
#    
#    rr, gg, bb = test_img.split()
#    
#    norm_rr = Image.fromarray(normalize(np.array(rr)), 'L')
#    norm_gg = Image.fromarray(normalize(np.array(gg)), 'L')
#    norm_bb = Image.fromarray(normalize(np.array(bb)), 'L')
#    
#    norm_img = Image.merge("RGB", (norm_rr, norm_gg, norm_bb))
#    norm_img.save('normalized_image.jpg')
#    
#    quan_rr = quantize(normalize(np.array(rr)))
#    quan_gg = quantize(normalize(np.array(gg)))
#    quan_bb = quantize(normalize(np.array(bb)))
#
#    quan_img = Image.merge("RGB", (quan_rr, quan_gg, quan_bb))
#    quan_img.save('quantized_image.jpg')
#    plot_hist(quan_img.histogram(),"quantized_image_histogram")
#    
#    ## exercise 4
#    width = 500
#    height = 300
#    path_images = parse_dataset_dir('/home/lpineda/deep-learning-course/test/words')
#    #dataset = read_dataset(path_images, width, height)
#    
#    test_img =  read_image(path_images[19])
#    test_img.save('words_image.jpg')
#    plot_hist(test_img.histogram(),"words_image_histogram")
#
#    
#    rr, gg, bb = test_img.split()
#    
#    quan_rr = quantize(normalize(np.array(rr)))
#    quan_gg = quantize(normalize(np.array(gg)))
#    quan_bb = quantize(normalize(np.array(bb)))
#
#    quan_img = Image.merge("RGB", (quan_rr, quan_gg, quan_bb))
#    quan_img.save('words_quantized_image.jpg')
#    plot_hist(quan_img.histogram(),"words_quantized_image_histogram")

    
    ## exercise 5
    sentences = get_sentences("nietzsche.txt")
    words = split_sentences(sentences)    
    print("There are {0} words in {1} sentences".format(len(words),len(sentences)))
    
    word_count_sorted = count_words_frequency(words)
    print("There are {0} different words".format(len(word_count_sorted)))
        
    top1000words_vectors = map(lambda x: map_frequency(x, word_count_sorted[0:100]), sentences)
    
    pca_top1000words = PCA(np.array(top1000words_vectors))

    print pca_top1000words
    #print(pca_top1000words)
    #low1000words_vectors = map(lambda x: map_frequency(x, word_count_sorted[-1000:], inverse=True), sentences)

    #print low1000words_vectors
    #low1000vec = map (lambda x: np.hstack((x, [0]*(max_top1000-len(x)))), low1000words)
    #pca_low1000 = PCA(np.array(low1000vec))
    

if __name__ == "__main__":
    main()
    