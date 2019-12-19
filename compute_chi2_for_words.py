from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np


import argparse

parser = argparse.ArgumentParser(description='Computes the chi2 value for words present in our vocabulary.'+
                                             'The chi2 value may be Nan!')

parser.add_argument('--df_cleaned_sentences',
                    required=True,
                    help='dataframe with cleaned sentences')

parser.add_argument('--output_pickle',
                    required=True,
                    help='pickle where the chi2 dataframe will be saved')

args = parser.parse_args()

df = pd.read_pickle(args.df_cleaned_sentences)

# Extract all unique words in the sentences
vocabulary = df['sentence'].apply(lambda x: x.split()).explode().unique()

print("The vocabulary size is {}".format(len(vocabulary)))

# Compute the chi2 score, using the same vocabulary as provided by the dataframe
vectorizer = CountVectorizer(lowercase=True,stop_words='english', vocabulary=vocabulary)
X = vectorizer.fit_transform(df.sentence)
chi2score = chi2(X, df.label)[0]

# Create a dataframe with words and chi2 score.
df_chi2 = pd.DataFrame({"word": vectorizer.get_feature_names(), "chi2": chi2score})

print("The number of nan entries is {}".format(len(df_chi2[df_chi2.chi2.isna()])))

# Save it to file.
df_chi2.to_pickle(args.output_pickle)