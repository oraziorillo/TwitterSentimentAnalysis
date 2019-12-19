from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

# For command line argument parsing
import argparse

from helpers import count_unique_words, count_unique_ngrams, \
            build_unique_ngrams, create_sentence_vectors, create_sentence_vectors_submission

import sys

from clean_helpers import *

from data_handling import build_sentences

from timeit import default_timer as timer


parser = argparse.ArgumentParser(description='This file creates a dataframe of cleaned sentences')

parser.add_argument('--input_pos',
                    required=True,
                    help='Input file with positive tweets')
parser.add_argument('--input_neg',
                    required=True,
                    help='Input file with negative tweets')
parser.add_argument('--output_df_train',
                    required=True,
                    help='Pickle file with df cleaned')
parser.add_argument('--clean_methods', 
                    required=True,
                    help='Cleaning methods, can choose among: \n' + 
                        "clean_new_line, " +
                        "lowercase, " + "lemmatize (textBlob one), remove_stopwords, " +
                        "clean_punctuation, clean_tags, remove_numbers, " +
                        "remove_saxon_genitive, more_than_double_rep, " +
                        "remove_@, remove_urls, " 
                        "lemmatize_spacy (better to use either this either textblob one, not both), ",
                    nargs='+')

# If not provided, then it just doesn't clean words according to the vocabulary of the 
# chosen model.
parser.add_argument('--model_word_embedding',
                    required=False,
                    help='Name of the model to extract the vocabulary, can either be w2v or glove.'+
                         ' If not provided, then all words are kept and no further cleaning is applied. '+
                         'Make sure you have the models for glove and word2vec in the following files: '+
                         'models/glove.twitter.27B.200d.txt and models/GoogleNews-vectors-negative300.bin')


args = parser.parse_args()

cleaning_options = args.clean_methods

# Cleaning methods defined in clean_helpers.py
clean = {
    "clean_new_line": clean_new_line,
    "lowercase": lowercase,
    "lemmatize": lemmatize,
    "remove_stopwords": remove_stopwords,
    "clean_punctuation": clean_punctuation,
    "clean_tags" : clean_tags,
    "remove_numbers": remove_numbers,
    "remove_saxon_genitive": remove_saxon_genitive,
    "more_than_double_rep": clean_more_than_double_repeated_chars,
    "clean_spelling": clean_spelling,
    "lemmatize_spacy": lemmatize_spacy,
    "remove_@": remove_ats,
    "remove_urls": remove_urls
}

# File with positive tweets
input_file_pos = args.input_pos

# File with negative tweets
input_file_neg = args.input_neg
    
# Make a list of positive and negative sentences.
list_of_pos_sentences = []
with open(input_file_pos, 'r') as f:
    for line in f:
        list_of_pos_sentences.append(line)
 
list_of_neg_sentences = []
with open(input_file_neg, 'r') as f:
    for line in f:
        list_of_neg_sentences.append(line)

# Build the dataframe with columns: sentence, label (either 1 or -1)
df = build_sentences(list_of_pos_sentences, list_of_neg_sentences)

print("unique words = {}".format(count_unique_words(df)))

# Perform all the cleaning options selected
for clean_option in cleaning_options:
    counter_of_occurrences = 0
    start = timer()
    df = clean[clean_option](df)
    end = timer()
    print("Time elapsed: {}".format(end - start)) 
    print(clean_option)
    print(df.head())
    print("unique words = {}".format(count_unique_words(df)))
    print("################################\n\n")
    
if 'model_word_embedding' in args:
    # If we want to provide a model and filter all words which are not in the 
    # model's vocabulary:
    if args.model_word_embedding == 'w2v':
        # load the w2v model
        print("Clean words not present in the Word2Vec vocabulary")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        vocabulary = w2v_model.wv.vocab
    else:
        # load the glove model
        print("Clean words not present in the GloVe vocabulary")
        vocabulary = []
        with open('models/glove.twitter.27B.200d.txt') as f:
            for line in f:
                word, *vector = line.split()
                vocabulary.append(word)
    # Make a set for the vocabulary, otherwise operation `in` takes O(n) and not O(1)    
    vocabulary = set(vocabulary)

    # Delete all words that are not present in the vocabulary of the model
    df = clean_with_vocabulary_of_model(df, vocabulary)

print("unique words = {}".format(count_unique_words(df)))
df.to_pickle(args.output_df_train)

print("df pickle file correctly saved in {}".format(args.output_df_train))