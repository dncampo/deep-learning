import sys
sys.path.append("..")

from cifasis.dataset import *
from cifasis.imgfun import *

from sklearn import decomposition
import pylab as pl

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
        
    top1000words_vectors = map(lambda x: map_frequency(x, word_count_sorted[0:1000]), sentences)
    pca_top1000words = decomposition.PCA(n_components=2)
    pca_top1000words.fit(np.array(top1000words_vectors))
    X_top = pca_top1000words.transform(np.array(top1000words_vectors))
    pl.figure(1)
    pl.scatter(X_top[:, 0], X_top[:, 1])
    pl.savefig('pca_top_freq.svg', format='svg', dpi=1200)
    print("There are {0} top frequency PCA points.".format(len(X_top[:, 0])))

    low1000words_vectors = map(lambda x: map_frequency(x, word_count_sorted[-1000:], inverse=True), sentences)
    pca_low1000words = decomposition.PCA(n_components=2)
    pca_low1000words.fit(np.array(top1000words_vectors))
    X_low = pca_low1000words.transform(np.array(low1000words_vectors))
    pl.figure(2)
    pl.scatter(X_low[:, 0], X_low[:, 1])
    pl.savefig('pca_low_freq.svg', format='svg', dpi=1200)
    print("There are {0} low frequency PCA points.".format(len(X_low[:, 0])))
    

if __name__ == "__main__":
    main()
    