from nltk import ngrams
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

def create_labelled_file(name_file, train):
    """
    Create the input file for fasttext
    it is made using the __label__ prefix for 1, -1 (labels for :), :( )
    and the sentence itself
    """
    with open(name_file, 'w') as f:
        for index, row in train.iterrows():
            f.write("__label__{} {}\n".format(row['label'], row['sentence']))
            
    # A bit useless, but may be clearer
    return name_file


def count_unique_words(df):
    """
    counts the number of unique words in the vocabulary
    """
    df_list = df.sentence.apply(lambda x: x.split())
    df_list_exploded = df_list.explode()
    return len(df_list_exploded.unique())


def count_unique_ngrams(df, n):
    """
    Count the number of ngrams
    """
    return len(df.sentence.apply(lambda x: [x for x in ngrams(x.split(), n)]).explode().unique())


def build_unique_ngrams(df, n):
    """
    Count the number of ngrams
    """
    unique_ngrams = df['sentence'].apply(lambda x: [x for x in ngrams(x.split(), n)]).explode().unique()
    
    new_unique_ngrams = []
    for el in unique_ngrams:
        if isinstance(el, tuple):
            string = ""
            for item in el:
                string += item + " "
            new_unique_ngrams.append(string[:-1])
    
    return new_unique_ngrams


def create_sentence_vectors(X, Y, word_vector_size, w2v_model):
    """
    X must be a vector of sentences
    Y must be a vector of labels (1, 0)
    word_vector_size is the size of the word vector (100-1000)
    w2v model is the Word2Vec model trained in advance.
    
    the returned sentence_y and sentence_x may be shorter than X and Y.
    This can happen in case no word in X is in our vocabulary.
    (Only when it is computed the test set!)
    
    :returns sentence_x: a numpy array, with 1 array of size size_of_wordvector,
                obtained by averaging the word vectors of every word in every sentence.
    :returns sentence_y: the labels, encoded as vectors [1,0] or [0,1]
    
    """
    counter_of_zero_sentences = 0
    sentence_x = []
    sentence_y = []
    for sent, label in tqdm(zip(X, Y)):
        sentence_vector = np.zeros(word_vector_size) # Probably most common word, we should always find it
        words_in_vocabulary = 0
        for word in sent.split():
            if word in w2v_model.wv.vocab:
                sentence_vector += w2v_model.wv[word]  # wc[] is a numpy vector
                words_in_vocabulary += 1
        if words_in_vocabulary > 0:
            sentence_x.append(sentence_vector / words_in_vocabulary)   # Take the average
            sentence_y.append(label)
        else:
            counter_of_zero_sentences += 1
            sentence_x.append(sentence_vector)   # Do some pretty bad inference: append a zero vector. Maybe it is better to append the average of the column, or the median
            sentence_y.append(label)
    sentence_x = np.array(sentence_x)
    
    sentence_y = to_categorical(np.array(sentence_y))
    
    print("the number of zero sentences (the sentences which have 0 words in our vocabulary) is {}".
         format(counter_of_zero_sentences))
    return sentence_x, sentence_y



def create_sentence_vectors_submission(X, word_vector_size, w2v_model):
    """
    X must be a vector of sentences
    Y must be a vector of labels (1, 0)
    word_vector_size is the size of the word vector (100-1000)
    w2v model is the Word2Vec model trained in advance.
    
    the returned sentence_y and sentence_x may be shorter than X and Y.
    This can happen in case no word in X is in our vocabulary.
    (Only when it is computed the test set!)
    
    :returns sentence_x: a numpy array, with 1 array of size size_of_wordvector,
                obtained by averaging the word vectors of every word in every sentence.
    :returns sentence_y: the labels, encoded as vectors [1,0] or [0,1]
    
    """
    counter_of_zero_sentences = 0
    sentence_x = []
    for sent in X:
        sentence_vector = np.zeros(word_vector_size) # Probably most common word, we should always find it
        words_in_vocabulary = 0
        for word in sent.split():
            if word in w2v_model.wv.vocab:
                sentence_vector += w2v_model.wv[word]  # wc[] is a numpy vector
                words_in_vocabulary += 1
        if words_in_vocabulary > 0:
            sentence_x.append(sentence_vector / words_in_vocabulary)   # Take the average
        else:
            counter_of_zero_sentences += 1
            sentence_x.append(sentence_vector)   # Do some pretty bad inference: append a zero vector. Maybe it is better to append the average of the column, or the median
    sentence_x = np.array(sentence_x)
    
    
    print("the number of zero sentences (the sentences which have 0 words in our vocabulary) is {}".
         format(counter_of_zero_sentences))
    return sentence_x