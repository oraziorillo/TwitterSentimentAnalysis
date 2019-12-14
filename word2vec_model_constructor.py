import pickle
import pandas as pd
import numpy as np
import argparse

import gensim   # Not sure whether it is better to use gensim or tensorflow :/
import logging
from gensim.models.phrases import Phrases, Phraser

import multiprocessing

from gensim.models import Word2Vec

parser = argparse.ArgumentParser(description='Builds the word vector model')

parser.add_argument('--train_pickle',
                    required=True,
                    help='input pickle file (the training set)')

parser.add_argument('--word_vector_size',
                    help='The size of the representation space for words (100-1000 advised)',
                    required=True)

parser.add_argument("--sg",
                    help="Use skip gram or continuous bags of words, either 1 or 0 (usually sg works better for larger datasets)",
                    required=True)

parser.add_argument("--window",
                    help="window of the word2vec model (better 5 for cbow or 10 for sg according to authors)")

parser.add_argument('--epochs',
                    help='The number of epochs the ',
                    required=True)

parser.add_argument('--output_model',
                    help='The output file for the word2vec model',
                    required=True)

                

args = parser.parse_args()

df = pd.read_pickle(args.train_pickle)

epochs = int(args.epochs)   # The number of epochs the word2vec model will be trained for.
window = int(args.window)
sg = int(args.sg) == 1



# Create train_x and test_x sets (shuffle and do split)
train_test_split = 0.8
permut = np.random.permutation(df.shape[0])
train_x = df.iloc[permut[: int(df.shape[0]*train_test_split)]]['sentence']
train_y = df.iloc[permut[: int(df.shape[0]*train_test_split)]]['label']
test_x = df.iloc[permut[int(df.shape[0]*train_test_split): ]]['sentence']
test_y = df.iloc[permut[int(df.shape[0]*train_test_split): ]]['label']

# convert -1 labels into 0
train_y = train_y.where(train_y == 1, 0) 
test_y = test_y.where(test_y == 1, 0)

sentences = [row.split() for row in train_x]
len(sentences)

word_vector_size = int(args.word_vector_size)   # should be among 100-1000

# logging.root.level = logging.ERROR   # Should reduce logging

w2v_model = Word2Vec(min_count=1,
                     window=10,      # Advised by Authors when using skip-gram
                     size=word_vector_size,
                     negative=5,
                     workers=4,
                     sg=1)    ## Careful here: it should work better with sg=1 for big data

# Build the vocabulary (not sure why it is needed, but still)
w2v_model.build_vocab(sentences, progress_per=100000)

# Train the model.
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=epochs)

# store the model to file.
w2v_model.save(args.output_model)