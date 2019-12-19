
import numpy as np
import matplotlib.pyplot as plt
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
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
sys.path.append('../')

import argparse

parser = argparse.ArgumentParser(description='Trains a dense neural network to predict'+
                                            'the labels of the tweets')

parser.add_argument('--epochs',
                    required=True,
                    help='numpy array train sentence embeddings')

parser.add_argument('--np_sentences_train_x',
                    required=True,
                    help='numpy array train sentence embeddings')

parser.add_argument('--np_sentences_train_y',
                    required=True,
                    help='numpy array train categorical labels')

parser.add_argument('--np_sentences_test_x',
                    required=False,
                    help='numpy array test sentence embeddings ()')

parser.add_argument('--np_sentences_test_y',
                    required=False,
                    help='numpy array test categorical labels ()')

parser.add_argument('--test_locally',
                    required=True,
                    help='Either 1 or 0, if set to 0 save the model at the end, and you don\'t need to provide the validation (test) arrays')

parser.add_argument('--layer_depth',
                    required=True,
                    help='The number of layers of the neural network')

parser.add_argument('--layer_size',
                    required=True,
                    help='The layer size for the neural network')
                    
parser.add_argument('--filepath_model',
                    required=True,
                    help='File where the best model will be saved')

args = parser.parse_args()

sentence_train_x = np.load(args.np_sentences_train_x)
sentence_train_y = np.load(args.np_sentences_train_y)

if args.test_locally == '1':
    # We don't want to submit, then we have the validation data. 
    sentence_test_x = np.load(args.np_sentences_test_x)
    sentence_test_y = np.load(args.np_sentences_test_y)

word_vector_size = sentence_train_x.shape[1]  # The size of the sentence (and word) embedding

from tensorflow.keras.callbacks import ModelCheckpoint
# now perform training on the new features vectors.

# Build a "deep" neural network with 2 hidden layers. When we see that it somehow works,
# we can start doing some cross validation on it.


model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(sentence_train_x.shape[1],)))

for i in range(int(args.layer_depth)):
    # Add layers to the model
    model.add(keras.layers.Dense(int(args.layer_size), activation='relu')) 

model.add(keras.layers.Dense(2, activation='softmax'))   # Only 0 and 1)


model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )

filepath=args.filepath_model
if args.test_locally == '1':
    # We are just testing locally.
    # Therefore we create a callback to save the model whenever we improve the test 
    # accuracy.
    checkpoint = ModelCheckpoint(filepath, 
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='max')
    callbacks_list = [checkpoint]

    model.fit(x=sentence_train_x,
                y=sentence_train_y,
                validation_data=(sentence_test_x,  sentence_test_y),
                callbacks=callbacks_list,
                epochs=int(args.epochs),
                use_multiprocessing=True,
                batch_size=128)

else:
    # Just train for the epochs passed as parameters, and save the model at the end.
    model.fit(x=sentence_train_x,
                y=sentence_train_y,
                epochs=int(args.epochs),
                use_multiprocessing=True,
                batch_size=128)
    print("Save the model in {}".format(filepath))
    model.save_weights(filepath)
