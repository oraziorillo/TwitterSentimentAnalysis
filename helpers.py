import pandas as pd

from nltk import ngrams
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# -------------------------------------------------------------------------------------------------------------------
# Bert stuff
# -------------------------------------------------------------------------------------------------------------------

def help_bert_sentence_embeddings():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 bert_sentence_embeddings.py <sentences_to_embed.csv>")
    

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


# -------------------------------------------------------------------------------------------------------------------
# LSTM stuff
# -------------------------------------------------------------------------------------------------------------------

def help_train_lstm():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 help_train_lstm.py <number_of_epochs>")
    
def help_keep_training_lstm():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 help_keep_training_lstm.py <model_name> <number_of_epochs>")
    
def help_generate_predictions():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 generate_predictions.py <model_name>")


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