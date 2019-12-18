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
                    help='input file with positive tweets')
parser.add_argument('--input_neg',
                    help='Input file with negative tweets')
parser.add_argument('--output_df_train',
                    help='Pickle file with df cleaned')
parser.add_argument('--clean_methods', 
                    help='cleaning methods, can choose among: \n' + 
                        "clean_new_line, " +
                        "lowercase, " + "lemmatize (textBlob one), remove_stopwords, " +
                        "clean_punctuation, clean_tags, remove_numbers, " +
                        "remove_saxon_genitive, gensim_simple, more_than_double_rep, " +
                        "remove_@, remove_urls" 
                        "lemmatize_spacy (better to use either this either textblob one, not both)",
                    nargs='+')


args = parser.parse_args()

print(args.clean_methods)

take_full = True
test_locally = True
create_new_text_files = True
ngrams = 1

# Specify here what cleaning functions you want to use
cleaning_options = args.clean_methods

# Cleaning methods defined in clean_helpers.py
clean = {
    "clean_new_line": clean_new_line,
    "lowercase": lowercase,
    "lemmatize": lemmatize,
    "remove_stopwords": remove_stopwords,
    "translate": perform_translation,
    "clean_punctuation": clean_punctuation,
    "clean_tags" : clean_tags,
    "remove_numbers": remove_numbers,
    "remove_saxon_genitive": remove_saxon_genitive,
    "gensim_simple": gensim_clean,   # not a good idea to use it I think! It cleans everything which is not alphabetic (special char, numbers and so on)
    "more_than_double_rep": clean_more_than_double_repeated_chars,
    "clean_spelling": clean_spelling,
    "lemmatize_spacy": lemmatize_spacy,
    "remove_@": remove_ats,
    "remove_urls": remove_urls
}


input_file_pos = args.input_pos

input_file_neg = args.input_neg
    
list_of_pos_sentences = []
with open(input_file_pos, 'r') as f:
    for line in f:
        list_of_pos_sentences.append(line)
 
list_of_neg_sentences = []
with open(input_file_neg, 'r') as f:
    for line in f:
        list_of_neg_sentences.append(line)

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
    
df.head()

df.to_pickle(args.output_df_train)

print("df pickle file correctly saved in {}".format(args.output_df_train))