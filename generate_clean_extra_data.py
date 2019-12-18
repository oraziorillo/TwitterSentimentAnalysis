import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from clean_helpers import *
from tqdm.notebook import tqdm
from clean_helpers import *
from helpers import count_unique_words, count_unique_ngrams, \
            build_unique_ngrams, create_sentence_vectors, create_sentence_vectors_submission


clean = {
    "clean_new_line": clean_new_line,
    "lowercase": lowercase,
    "lemmatize": lemmatize,
    "remove_stopwords": remove_stopwords,
    "translate": perform_translation,
    "clean_punctuation": clean_punctuation,
    "clean_tags": clean_tags,
    "remove_numbers": remove_numbers,
    "remove_saxon_genitive": remove_saxon_genitive,
    "gensim_simple": gensim_clean,   # not a good idea to use it I think! It cleans everything which is not alphabetic (special char, numbers and so on)
    "more_than_double_rep": clean_more_than_double_repeated_chars,
    "clean_spelling": clean_spelling,
    "lemmatize_spacy": lemmatize_spacy,
    "remove_ats": remove_ats,
    "remove_urls": remove_urls,
    "remove_ampersand": remove_ampersand,
    "clean_empty_sentences": clean_empty_sentences
}

cleaning_options = ['clean_new_line', 'lowercase', 'clean_punctuation_2', 'clean_tags', 'remove_numbers', 
                    'lemmatize_spacy', 'more_than_double_rep', 'clean_empty_sentences']
                    
                    
df = pd.read_csv("Data/extra_data.csv", usecols=range(4))

df['sentence'] = df['SentimentText']
df = df.drop(["SentimentText"], axis=1)
df['label'] = df['Sentiment']
df = df.drop(["Sentiment"], axis=1)

%time df_cleaned = tokenize_tweets(df)

# Perform all the cleaning options selected
for clean_option in cleaning_options:
    %time df_cleaned = clean[clean_option](df_cleaned)
    print(clean_option)
    print("unique words = {}".format(count_unique_words(df_cleaned)))
    print("################################\n\n")
    
df_cleaned.to_csv("Data/clean_extra_data.csv")