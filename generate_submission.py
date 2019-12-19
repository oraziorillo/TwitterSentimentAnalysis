import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

from helpers import count_unique_words, count_unique_ngrams, \
            build_unique_ngrams, create_sentence_vectors, create_sentence_vectors_submission

import sys
import re

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


parser = argparse.ArgumentParser(description='Generate the submission csv')

parser.add_argument('--df_test_cleaned',
                    required=True,
                    help='dataframe with test set previously cleaned.')

parser.add_argument('--word2vec_model',
                    required=True,
                    help='The filepath for the word2vec model')

parser.add_argument('--layer_depth',
                    required=True,
                    help='The number of layers of the neural network')

parser.add_argument('--layer_size',
                    required=True,
                    help='The layer size for the neural network')
                    
parser.add_argument('--filepath_model',
                    required=True,
                    help='File path for the neural net model') 

parser.add_argument('--submission_path',
                    required=True,
                    help='The Submission csv file path') 

args = parser.parse_args()

if re.match(r".*GoogleNews-vectors-negative300", args.word2vec_model) != None:
    # This regex is used because, I think due to different formats, Google's 
    # pre-trained model must be loaded in a different way than what you would do 
    # normally
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_model, binary=True)
else:
    # It's a normal Word2Vec model
    w2v_model = Word2Vec.load(args.word2vec_model)

word_vector_size = w2v_model.wv.vectors.shape[1]

test_df_cleaned = pd.read_pickle(args.df_test_cleaned)

sentence_submission_x = create_sentence_vectors_submission(test_df_cleaned['sentence'],
 word_vector_size, w2v_model)

# Recreate the model that achieved the best submission
model_star = keras.Sequential()
model_star.add(keras.layers.InputLayer(input_shape=(sentence_submission_x.shape[1],)))
for i in range(int(args.layer_depth)):
    model_star.add(keras.layers.Dense(int(args.layer_size), activation='relu'))

model_star.add(keras.layers.Dense(2, activation='softmax'))   # Only 0 and 1)

model_star.load_weights(args.filepath_model)

model_star.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Compute the predictions
model_star.predict(sentence_submission_x)

predictions = []
for el in model_star.predict(sentence_submission_x):
    predictions.append(-1 if el[0] > el[1] else 1)

# Show some predictions to see if they respect the format -1, 1
print(predictions[:10])

results = pd.DataFrame({
    "Id": test_df_cleaned['label'],
    "Prediction": predictions
})

# Show some results to see if they make sense.
print(results.head(20))

# Generate the submission file
results.to_csv(args.submission_path, index=False)

print("csv correctly saved in {}".format(args.submission_path))