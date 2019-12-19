import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import re
import multiprocessing

from helpers import count_unique_words, count_unique_ngrams, \
            build_unique_ngrams, create_sentence_vectors, create_sentence_vectors_submission

import sys

import gensim
import logging
from gensim.models.phrases import Phrases, Phraser

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

parser.add_argument("--limit",
                    required=False,
                    help="if provided, the number of sentences of df to consider (may be unfeasable to take them all)")

parser.add_argument('--output_np_y',
                    required=True,
                    help='numpy array with labels represented as categorical')

args = parser.parse_args()


if re.match(r".*GoogleNews-vectors-negative300", args.w2v_model) != None:
    # This regex is used because, I think due to different formats, Google's 
    # pre-trained model must be loaded in a different way than what you would do 
    # normally
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(args.w2v_model, binary=True)
else:
    # It's a normal Word2Vec model
    w2v_model = Word2Vec.load(args.w2v_model)

word_vector_size = w2v_model.wv.vectors.shape[1]

train_df = pd.read_pickle(args.df_cleaned_sentences) 

# If the argument limit is not provided, just take all sentences
limit = int(args.limit) if args.limit is not None else train_df.shape[0]

x = train_df.iloc[:limit]['sentence']
y = train_df.iloc[:limit]['label']

#########################################
### Super important here ################
#########################################

# We need to set labels from the -1, 1 representation into the 0,1 representation
y = y.where(y == 1, 0) 

# Create sentence vectors
sentence_x, sentence_y = create_sentence_vectors(x, y, word_vector_size, w2v_model)

np.save(args.output_np_x, sentence_x)

np.save(args.output_np_y, sentence_y)

print("vectors saved")




