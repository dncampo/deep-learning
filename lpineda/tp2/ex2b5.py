from keras.preprocessing.text import *

f = open('nietzsche.txt', 'r')
strings = f.read()
words = text_to_word_sequence(strings, filters=base_filter() + "1234567890", lower=True, split=" ")

sentences = text_to_word_sequence(strings, filters="", lower=False, split='\n')
sentences = text_to_word_sequence(" ".join(sentences), filters="1234567890" , lower=True, split='.')
sentences = list(map(lambda x: ''.join(c for c in x if c not in base_filter()), sentences))

print('\n'.join(sentences))

tokenizer = Tokenizer(nb_words=1000)
tokenizer.fit_on_texts(words)

#m is a numpy matrix with the encoded sentences
m = tokenizer.texts_to_matrix(sentences, mode='count')
