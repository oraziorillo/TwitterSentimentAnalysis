import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from helpers import count_unique_words, count_unique_ngrams, \
            build_unique_ngrams, create_sentence_vectors, create_sentence_vectors_submission

import sys

import gensim   # Not sure whether it is better to use gensim or tensorflow :/
import logging
from gensim.models.phrases import Phrases, Phraser

import multiprocessing

from gensim.models import Word2Vec
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
sys.path.append('../')


import argparse

parser = argparse.ArgumentParser(description='Builds sentence representation using word vectors.')

parser.add_argument('--w2v_model',
                    required=True,
                    help='Word2Vec model pretrained')

parser.add_argument('--df_cleaned_sentences',
                    required=True,
                    help='dataframe with cleaned sentences')

parser.add_argument('--output_np_x',
                    required=True,
                    help='numpy array with cleaned sentences represented as vectors')

parser.add_argument('--output_np_y',
                    required=True,
                    help='numpy array with labels represented as categorical')

args = parser.parse_args()

w2v_model = Word2Vec.load(args.w2v_model)

word_vector_size = w2v_model.wv.vectors.shape[1]

train_df = pd.read_pickle(args.df_cleaned_sentences) 

x = train_df['sentence']
y = train_df['label']

#########################################
### Super important here ################
#########################################

# We need to set labels from the -1, 1 representation into the 0,1 representation
y = y.where(y == 1, 0) 


sentence_x, sentence_y = create_sentence_vectors(x, y, word_vector_size, w2v_model)

np.save(args.output_np_x, sentence_x)

np.save(args.output_np_y, sentence_y)




