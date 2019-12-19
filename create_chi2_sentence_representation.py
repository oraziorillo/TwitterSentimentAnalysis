import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import re
import multiprocessing

from helpers import count_unique_words, count_unique_ngrams, \
            build_unique_ngrams, create_sentence_vectors, create_sentence_vectors_submission, \
            create_sentence_chi2_vectors

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

parser.add_argument('--df_chi2',
                    required=True,
                    help='dataframe with chi2 for every word in our vocabulary.')

parser.add_argument('--output_np_x',
                    required=True,
                    help='numpy array with cleaned sentences represented as vectors')

parser.add_argument("--limit",
                    required=False,
                    help="if provided, the number of sentences of df to consider (may be unfeasable to take them all) OPTIONAL")

parser.add_argument('--output_np_y',
                    required=True,
                    help='numpy array with labels represented as categorical')

args = parser.parse_args()


#### CAREFUL HERE
"""
Some chi2 values are Nan, probably as the words don't appear enough often to be 
in every category (0,0) (0,1) (1,0) (1,1)
(check Text classification and Naive Bayes to understand what the classes mean, 
https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf)
You can just drop those words, they are only 500 out of about 60k-100k words in the vocabulary.
"""


if re.match(r".*GoogleNews-vectors-negative300", args.w2v_model) != None:
    # Load google big word2vec model, dunno why but it is in a very strange format :/
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(args.w2v_model, binary=True)
else:
    # Load any other normal word2vec model
    w2v_model = Word2Vec.load(args.w2v_model)

train_df = pd.read_pickle(args.df_cleaned_sentences)   # Read the dataframe with sentences

chi2_df = pd.read_pickle(args.df_chi2)                  # Read the dataframe with chi2 values for every word in the vocabulary
chi2_df.set_index("word", inplace=True)   # Set the index to the words themselves (in this way we have "faster" lookup)

# limit is optional, very useful to make test if the data fits in memory!
limit = train_df.shape[0] if not args.limit else int(args.limit)     # Limit is to make tests with smaller sizes of the dataframe

x = train_df.iloc[:limit]['sentence']
y = train_df.iloc[:limit]['label']

#########################################
### Super important here ################
#########################################

# We need to set labels from the -1, 1 representation into the 0,1 representation
y = y.where(y == 1, 0) 

sentence_x, sentence_y = create_sentence_chi2_vectors(x, y, w2v_model, chi2_df)   ## Compute the sentence embedding

np.save(args.output_np_x, sentence_x)

np.save(args.output_np_y, sentence_y)




