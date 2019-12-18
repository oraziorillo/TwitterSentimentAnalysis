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
    sentence_x = np.empty( (len(X), word_vector_size) )  ## Initialize of the right dimension.
    # The initial value 0 is totally fake.
    sentence_y = np.empty(len(Y))
    # avg_vector = np.mean(X, axis=0)
    print("Computed the average vector")

    for i, (sent, label) in tqdm(enumerate(zip(X, Y))):
        # print(i)
        sentence_vector = np.zeros(word_vector_size) # Probably most common word, we should always find it
        words_in_vocabulary = 0
        for word in sent.split():
            if word in w2v_model.wv.vocab:
                sentence_vector += w2v_model.wv[word]  # wc[] is a numpy vector
                words_in_vocabulary += 1
        if words_in_vocabulary > 0:
            # print(sentence_vector / words_in_vocabulary)
            sentence_x[i] = np.array(sentence_vector / words_in_vocabulary)   # Take the average
            sentence_y[i] = label
        else:
            counter_of_zero_sentences += 1
            sentence_vector = np.zeros(word_vector_size)
            sentence_x[i] = sentence_vector    # Take the average
            sentence_y[i] = label
    
    sentence_y = to_categorical(sentence_y)
    
    print("the number of zero sentences (the sentences which have 0 words in our vocabulary) is {}".
         format(counter_of_zero_sentences))
    return sentence_x, sentence_y



def create_sentence_vectors_submission(X, word_vector_size, w2v_model):
    """
    X must be a vector of sentences
    word_vector_size is the size of the word vector (100-1000)
    w2v model is the Word2Vec model trained in advance.
    
    the returned sentence_y and sentence_x may be shorter than X and Y.
    This can happen in case no word in X is in our vocabulary.
    (Only when it is computed the test set!)
    
    :returns sentence_x: a numpy array, with 1 array of size size_of_wordvector,
                obtained by averaging the word vectors of every word in every sentence.
    :returns sentence_y: the labels, encoded as vectors [1,0] or [0,1]
    
    """

    avg_vector = np.mean(X, axis=0)

    counter_of_zero_sentences = 0
    sentence_x = np.array(len(X))
    for i, sent in enumerate(X):
        sentence_vector = np.zeros(word_vector_size) # Probably most common word, we should always find it
        words_in_vocabulary = 0
        for word in sent.split():
            if word in w2v_model.wv.vocab:
                sentence_vector += w2v_model.wv[word]  # wc[] is a numpy vector
                words_in_vocabulary += 1
        if words_in_vocabulary > 0:
            sentence_x[i] = (sentence_vector / words_in_vocabulary)   # Take the average
        else:
            sentence_vector = avg_vector.copy()   # Append the average vector, better than a vector of zeros!
            counter_of_zero_sentences += 1
            sentence_x[i] = (sentence_vector)   # Do some pretty bad inference: append a zero vector. Maybe it is better to append the average of the column, or the median
    
    print("the number of zero sentences (the sentences which have 0 words in our vocabulary) is {}".
         format(counter_of_zero_sentences))
    return sentence_x


def create_sentence_chi2_vectors(x, y, word_vector_size, w2v_model, chi2_df):
    """
    This method computes sentence vectors given the word embedding (in w2v_model)
    Instead of just taking the average, we take a weighted average by the 
    chi2 value of every word in the vocabulary.

    chi2 value could be Nan, in this case just skip the word.
    If a sentence doesn't have any valid word (either it is empty either its chi2 is Nan)
    then just put a random vector? Or a zero vector or the average (now we compute the average on a random amount of words
    taken randomly from 1k sentences).

    Hopefully, it doesn't happen too often

    :param x: the series with sentences
    :param y: the series with labels (must be converted into 0 1, not -1 1) and will be converted into categorical [0,1] [1,0]
    :param word_vector_size: the size of the word embedding
    :param w2v_model: the w2v model
    :param chi2_df: a dataframe which contains all the chi2 square values for each word

    :return sentence_x: a vector representation for each sentence, in the same dimension of word_vector_size
    :return sentence_y: the categorical labels for y
    """

    counter_of_zero_sentences = 0     # This counts how many times we have an empty sentence.
    sentence_x = np.empty( (len(x), word_vector_size) )  ## Initialize of the right dimension.
    
    # Here we compute an average word vector, so that we can use this when we have sentences with 0 valid words.
    # We could have substituted a zero vector, but I guess the mean makes less harm.
    # We compute the average by taking 1000 sentences, and averaging the word vectors of the words
    # appearing in the sentences.
    word_set_avg = []
    for sent in x[:1000]:
        for el in sent.split():
            word_set_avg.append(el)  # Create a long list of random words.

    word_avg_mat = np.empty((len(word_set_avg), word_vector_size))  
    for i, el in enumerate(word_set_avg):
        word_avg_mat[i] = w2v_model.wv[el]   # Create a matrix with the word embedding

    avg_vector = np.mean(word_avg_mat, axis=0) # Take the mean over the columns of the matrix.
    print("The average vector is")
    print(avg_vector)  
    
    sentence_y = np.array(y)

    for i, sent in tqdm(enumerate(x)):
        sentence_vector = np.zeros(word_vector_size)    # Initialize to zero
        words_in_vocabulary = 0
        tot_chi2 = 0
        for word in sent.split():
            if word in w2v_model.wv.vocab and not chi2_df.loc[word].isna()[0] :
                sentence_vector += (w2v_model.wv[word] * chi2_df.loc[word][0])  # we scale the word vector by its chi2 value
                words_in_vocabulary += 1
                tot_chi2 += chi2_df.loc[word][0]
        if words_in_vocabulary > 0:
            # We have at least one valid word in our sentence!
            sentence_x[i] = np.array(sentence_vector / tot_chi2)   # Take the weighted average by the chi2
        else:
            counter_of_zero_sentences += 1  # This can happen both if the sentence has no words, or if the word has no chi2 value.
            sentence_vector = avg_vector        # Now we take the average computed above.
            sentence_x[i] = sentence_vector    # Take the average or a zero vector! must decide which is best
    
    sentence_y = to_categorical(sentence_y)
    
    # print("the number of zero sentences (the sentences which have 0 words in our vocabulary) is {}".
    #      format(counter_of_zero_sentences))
    return sentence_x, sentence_y
