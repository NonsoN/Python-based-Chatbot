import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def sack_of_words(tokenized_sentence, all_words):
    '''EXAMPLE OF THE STRUCUTRE:
    sentence = ["hello", "what", "is", "your", "name"]
    words = ["hi", "hello", "what", "I", "you", "your", "first", "name", "last"]
    sack =   [0,     1,        1,     0,   0,     1,       0,       1,       0]

    You have the given sentence, the possible words, and then the sack which for each word
    either contains a 1(if the word is there) or a 0(if the word is absent)
    '''
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    #size of the length of the words, with a datatype of numpy.float32
    sack = np.zeros(len(all_words), dtype=np.float32)
    #loop over allwords
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            sack[idx] = 1.0

    return sack

'''sentence = ["hello", "what", "is", "your", "name"]
words = ["hi", "hello", "what", "I", "you", "your", "first", "name", "last"]
sack =sack_of_words(sentence, words)
print(sack)'''

'''x = "How long does it take to sign a player?"
print(x)
print(tokenize(x))
words = ["Organize", "updating", "tutorial"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)'''

