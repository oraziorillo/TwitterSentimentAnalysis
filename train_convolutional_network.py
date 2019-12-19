from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

from helpers import count_unique_words, count_unique_ngrams, \
            build_unique_ngrams, create_sentence_vectors, create_sentence_vectors_submission

import sys

import tensorflow as tf
from tensorflow import keras

import gensim   # Not sure whether it is better to use gensim or tensorflow :/
import logging
from gensim.models.phrases import Phrases, Phraser

import multiprocessing

from gensim.models import Word2Vec

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sys.path.append('../')

import argparse

parser = argparse.ArgumentParser(description='Builds sentence representation using word vectors.')

parser.add_argument('--w2v_model',
                    required=True,
                    help='Word2Vec model pretrained')

parser.add_argument('--filter_size',
                    nargs='+',
                    required=True,
                    help='a list of sizes for the convolutional filters (usually odd numbers. for example 3 5)')

parser.add_argument('--hidden_layers_size',
                    nargs='+',
                    required=True,
                    help='a list of sizes for the hidden layers (usually 50-100)')

parser.add_argument('--output_model',
                    required=True,
                    help='path where the model will be saved')
                    
args = parser.parse_args()

# Up to now everything is hardcoded, may be better to use parameters!

df = pd.read_pickle("dataframes/full_df_cleaned_train_0_8_glove200.pickle")

df_test = pd.read_pickle("dataframes/full_df_cleaned_test_0_2_glove200.pickle")

maxlen = 44    # magic number

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    counter_wrong = 0
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    for row in range(embedding_matrix.shape[0]):
        if not np.any(embedding_matrix[row,:]):
            counter_wrong += 1
            embedding_matrix[row,:] = np.random.rand(embedding_dim)

    print("The number of times we didn't find a word is {} and should be 0, wtf".format(counter_wrong))
    return embedding_matrix

def create_embedding_matrix_w2v(w2v_model, word_index):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    ## We can assume love is always present in our vocabulary ahaha
    embedding_matrix = np.zeros((vocab_size, w2v_model.wv.word_vec("love").shape[0]))  
    
    for word in w2v_model.wv.vocab:
        vector = w2v_model.wv.word_vec(word)
        if word in word_index:
            idx = word_index[word] 
            embedding_matrix[idx] = np.array(
                vector, dtype=np.float32)
    for row in range(embedding_matrix.shape[0]):
        if not np.any(embedding_matrix[row,:]):
            ### This should be checked again!!! Not sure it is correct!
            embedding_matrix[row,:] = np.random.rand(w2v_model.wv.vectors.shape[1])

    return embedding_matrix


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.sentence)

vocab_size = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(df.sentence)
X_test = tokenizer.texts_to_sequences(df_test.sentence)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

y_train = np.where(df.label == 1, 1, 0)
y_test = np.where(df_test.label == 1, 1, 0)

if args.w2v_model == 'w2v':
    # Use word2vec
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)

    embedding_matrix = create_embedding_matrix_w2v(
        w2v_model,
        tokenizer.word_index)

    ## Embedding dimension
    embedding_dim = w2v_model.wv.vectors.shape[1]
else:
    # Use glove
    embedding_dim = 200
    embedding_matrix = create_embedding_matrix(
        'glove/glove.twitter.27B.200d.txt',
        tokenizer.word_index, embedding_dim)


# Compile the model
from tensorflow.keras.layers import GlobalMaxPooling1D, concatenate, Dropout, Dense, Embedding, Input, Conv1D
from tensorflow.keras.models import Model

# Specifying the input shape: the input is a sentence of maxlen words
embedding_layer = Embedding(vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, 
                            trainable=True)
sequence_input = Input(shape=(maxlen,), dtype='int32')
# Creating the embedding using the previously constructed embedding matrix
embedded_sequences = embedding_layer(sequence_input)
convs = []
filter_sizes = [int(el) for el in args.filter_size]
for filter_size in filter_sizes:
    # Creating the convolutional layer:
    #    "filters" represents the number of different windows we want (i.e. how many channels to produce),
    #    therefore in our case we will end up with 200 different convolutions
    conv_layer = Conv1D(filters=256, 
                    kernel_size=filter_size, 
                    activation='relu')(embedded_sequences)
    # Creating the global max pooling layer
    pool_layer = GlobalMaxPooling1D()(conv_layer)
    convs.append(pool_layer)
merged_layers = concatenate(convs, axis=1)
# Create dropout leayer: randomly set a fraction of input units to 0, which helps prevent overfitting
x = Dropout(0.2)(merged_layers)  
# Create (regular) densely-connected layer
for el in args.hidden_layers_size:
    x = Dense(int(el), activation='relu')(x)
    x = Dropout(0.2)(x)

preds = Dense(1, activation='sigmoid')(x)
model_tw = Model(sequence_input, preds)
model_tw.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_tw.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

filepath="models/cnn_glove_tw"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Finally fit the model
history = model_tw.fit(X_train, y_train, epochs=15, verbose=True, validation_data=(X_test, y_test), callbacks=callbacks_list, batch_size=512)
loss, accuracy = model_tw.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model_tw.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)