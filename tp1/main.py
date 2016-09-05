import sys
sys.path.append("..")

from cifasis.dataset import *
from cifasis.imgfun import *

from matplotlib.mlab import PCA

def main():
    width = 500
    height = 500
    
    ## exercise 1
    path_images = parse_dataset_dir('/home/lpineda/101_ObjectCategories/laptop')
    #dataset = read_dataset(path_images, width, height)


    ## exercise 2
    test_img =  read_image(path_images[12])
    test_img.save('image.jpg')
    plot_hist(test_img.histogram(),"image_histogram")
    
    rr, gg, bb = test_img.split()
    
    norm_rr = Image.fromarray(normalize(np.array(rr)), 'L')
    norm_gg = Image.fromarray(normalize(np.array(gg)), 'L')
    norm_bb = Image.fromarray(normalize(np.array(bb)), 'L')
    
    norm_img = Image.merge("RGB", (norm_rr, norm_gg, norm_bb))
    norm_img.save('normalized_image.jpg')
    
    quan_rr = quantize(normalize(np.array(rr)))
    quan_gg = quantize(normalize(np.array(gg)))
    quan_bb = quantize(normalize(np.array(bb)))

    quan_img = Image.merge("RGB", (quan_rr, quan_gg, quan_bb))
    quan_img.save('quantized_image.jpg')
    plot_hist(quan_img.histogram(),"quantized_image_histogram")
    
    ## exercise 4
    width = 500
    height = 300
    path_images = parse_dataset_dir('/home/lpineda/deep-learning-course/test/words')
    #dataset = read_dataset(path_images, width, height)
    
    test_img =  read_image(path_images[19])
    test_img.save('words_image.jpg')
    plot_hist(test_img.histogram(),"words_image_histogram")

    
    rr, gg, bb = test_img.split()
    
    quan_rr = quantize(normalize(np.array(rr)))
    quan_gg = quantize(normalize(np.array(gg)))
    quan_bb = quantize(normalize(np.array(bb)))

    quan_img = Image.merge("RGB", (quan_rr, quan_gg, quan_bb))
    quan_img.save('words_quantized_image.jpg')
    plot_hist(quan_img.histogram(),"words_quantized_image_histogram")

    
    ## exercise 5
    sentences = get_sentences("/home/lpineda/deep-learning/tp1/nietzsche.txt")
    words = split_sentences(sentences)
    word_count = count_words_frequency(words)

#    top1000words = map(lambda x: map_sentence(x,dict(word_count[0:1000])), sentences)
#    top1000words = filter(lambda x: sum(x)!=0, top1000words)
#    low1000words = map(lambda x: map_sentence(x,dict(word_count[0:-1000])), sentences)
    ####################
#    max_top1000 = 0
#    for v in top1000words:
#        if len(v)>max_top1000:
#            max_top1000 = len(v)
#            
#    top1000array = map (lambda x: np.hstack((x, [0]*(max_top1000-len(x)))), top1000words)
#    top1000array = np.array(top1000array)
#    pca_top1000 = PCA(top1000array)
#    print(pca_top1000)
    
    #low1000vec = map (lambda x: np.hstack((x, [0]*(max_top1000-len(x)))), low1000words)
    #pca_low1000 = PCA(np.array(low1000vec))
    

if __name__ == "__main__":
    main()
