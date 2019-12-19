import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

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
    return len((df['sentence'].apply(lambda x: x.split())).explode().unique())


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
    :param X: list of sentences
    :param Y: list of labels (1, 0)
    :param word_vector_size: integer, the size of the word vector (100-1000)
    :param w2v_model: Word2Vec model
    
    the returned sentence_y and sentence_x may be shorter than X and Y.
    This can happen in case no word in X is in our vocabulary.
    
    :returns sentence_x: a numpy array of shape(len(X), word_vector_size)
                obtained by averaging the word vectors of every word in every sentence.
    :returns sentence_y: list of labels, encoded as vectors [1,0] or [0,1]
    """

    counter_of_zero_sentences = 0
    sentence_x = np.empty( (len(X), word_vector_size) )
    for i, sent in enumerate(X):
        sentence_vector = np.zeros(word_vector_size)
        words_in_vocabulary = 0
        for word in sent.split():
            if word in w2v_model.wv.vocab:
                sentence_vector += w2v_model.wv[word]  # wv[] is a numpy vector
                words_in_vocabulary += 1
        if words_in_vocabulary > 0:
            # If there is at least one word in our vocabulary
            sentence_x[i] = (sentence_vector / words_in_vocabulary)   # Take the average
        else:
            # If there is no word in our vocabulary
            # Append the zero vector. Could even use the average, but doesn't change much in practice.
            sentence_vector = np.zeros(word_vector_size)   
            counter_of_zero_sentences += 1
            sentence_x[i] = (sentence_vector)   # Do some pretty bad inference: append a zero vector. Maybe it is better to append the average of the column, or the median
    
    print("the number of zero sentences (the sentences which have 0 words in our vocabulary) is {}".
         format(counter_of_zero_sentences))
    return sentence_x


def create_sentence_chi2_vectors(x, y, w2v_model, chi2_df):
    """
    This method computes sentence vectors given the word embedding (in w2v_model)
    Instead of just taking the average, we take a weighted average by the 
    chi2 value of every word in the vocabulary.

    chi2 value could be Nan, in this case just skip the word.
    If a sentence doesn't have any valid word (either it is empty either its words' chi2 is Nan)
    then just put the average (now we compute the average on a random amount of words
    taken randomly from 1k sentences).

    :param x: pd.series with sentences
    :param y: pd.series with labels (must be converted into 0 1, not -1 1) and will be converted into categorical [0,1] [1,0]
    :param w2v_model: the w2v model
    :param chi2_df: a dataframe which contains all the chi2 square values for each word

    :return sentence_x: np.matrix of shape(len(x), embedding_size)
    :return sentence_y: the categorical labels for y [0,1] or [1,0] 
    """

    word_vector_size = w2v_model.wv.vectors.shape[1]
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

    word_avg_mat = np.empty((len(word_set_avg), word_vector_size))   # initialize an empty matrix 
    for i, el in enumerate(word_set_avg):
        word_avg_mat[i] = w2v_model.wv[el]   # Create the matrix with the word embeddings

    avg_vector = np.mean(word_avg_mat, axis=0)   # Take the mean over the columns of the matrix.
    
    sentence_y = np.array(y)

    for i, sent in tqdm(enumerate(x)):
        sentence_vector = np.zeros(word_vector_size)    # Initialize to zero
        words_in_vocabulary = 0
        tot_chi2 = 0
        for word in sent.split():
            if word in w2v_model.wv.vocab and word in chi2_df.index and not chi2_df.loc[word].isna()[0] :
                # If the word is in the vocabulary, and has a valid chi2 value
                sentence_vector += (w2v_model.wv[word] * chi2_df.loc[word][0])  # we scale the word vector by its chi2 value
                words_in_vocabulary += 1
                tot_chi2 += chi2_df.loc[word][0]
        if words_in_vocabulary > 0:
            # We have at least one valid word in our sentence!
            sentence_x[i] = np.array(sentence_vector / tot_chi2)   # Take the weighted average by the chi2
        else:
            counter_of_zero_sentences += 1  # This can happen both if the sentence has no words, or if the word has no chi2 value.
            sentence_vector = avg_vector        # We take the average computed above.
            sentence_x[i] = sentence_vector
    
    sentence_y = to_categorical(sentence_y)    # Convert to categorical
    
    print("the number of zero sentences (the sentences which have 0 words in our vocabulary) is {}".
         format(counter_of_zero_sentences))
    return sentence_x, sentence_y


def write(prediction, output_file):
    pd.DataFrame({
        "Id": range(1, len(prediction) + 1),
        "Prediction": prediction        
    }).to_csv(output_file, index=False)

    