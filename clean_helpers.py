import re
import pandas as pd
from textblob import TextBlob, Word
import nltk
import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer   # This lemmatizer I hope works better than the TextBlob one.

import spacy  # For lemmatization.

from nltk.tokenize import TweetTokenizer  # This is used to tokenize the second dataset.

def clean_tags(df):
    """
    remove all <...> occurrences
    """
    tag = "^<.*>$"
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x:  " ".join(" ".join([ w if w != "" and re.search(tag, w) is None else "" for w in x.split(" ")]).split())),
        'label': df.label
    })
    

def lowercase(df):
    """
    lowercase things
    """
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join([i.lower() for i in x.split(" ")])),
        "label": df.label
    })

def clean_new_line(df):
    """
    remove all \n
    Careful here: are we sure that we have no \n and whitespace together?
    We replace all occurrences of double whitespace with a single space!
    """
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join((" ".join(x.split("\n")[:])).split())),  # We replace all occurrences of 2 and 3 whitespace with 1 whitespace!
        "label": df.label    
    })


def clean_punctuation(df):
    """
    remove all punctuation and special characters
    """
    
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '/', '[', ']', "'", '@', '"', '\\', '-', '&', '*', '+', '#', '|', '^', '~'}
    
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([ "" if i in special_characters else i
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })

def remove_saxon_genitive(df):
    """
    remove all 's occurrences
    """
    
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([ i.replace("'s", "")
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })
    


def remove_stopwords(df):
    stopw = set(stopwords.words('english'))
    print("The number of scipy stopwords is {}".format(len(stopw)))

    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([ "" if i in stopw else i
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })


def remove_numbers(df):
    """
    removes numbers, they are useless
    """
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([el if re.search("^([-\.,/_\(\):x]?[0-9]+[-\.,/_\(\):x]?)+$", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })



def lemmatize(df):
    """
    Lemmatize tokens
    """
    nltk.download('wordnet')
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join([ Word(w).lemmatize() for w in x.split(" ")])),
        "label": df.label
    })


def lemmatize_spacy(df):
    """
    Do lemmatization with spacy library, looks like it is working better than
    TextBlob. Only thing, must be very careful with hashtags (they get split.)
    Probably better doing some more eye check controls to see if everything is
    OK. Other strange fact, it substitutes every pronoun as -PRON-
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])   # Load the spacy nlp module, in english.
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]).replace("# ", "#")),
        "label": df.label
    })


def gensim_clean(df):
    """
    Do gensim simple_preprocess.
    Performs a lot of cleaning 
    """    
    
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join(gensim.utils.simple_preprocess(x, min_len=0, max_len=500))),
        "label": df.label
    })
    
def reduce_lengthening(text):
    """
    This helper function takes any word with more than 2 repetitions of the same 
    char (yaaaay for instance, has 4 a repeated), and returns the word with at most
    2 repetitions.
    Can be useful to make similar words fall together:
    aaaaaah and aaah will both become aah for instance
    
    This comes from the fact that no english word has more than 2 characters repeated
    together.
    
    ccciiiiiioooo will be cciioo (all occurrences of repeated chars will be truncated to 2)
    
    Code is taken from https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def clean_more_than_double_repeated_chars(df):
    """
    Do gensim simple_preprocess.
    Performs a lot of cleaning 
    """    
    
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join([reduce_lengthening(w) for w in x.split()]) ),
        "label": df.label
    })


def clean_spelling(df):
    """ 
    Still need to finish it, it has problems as
    spelling.suggest yelds a generator, not a proper list.
    """
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join([spelling.suggest(w)[0][0] if spelling.suggest(w)[0][1] > 0.99 else w for w in x.split()]) ),
        "label": df.label
    })

    
def tokenize_tweets(df):
    tknzr = TweetTokenizer()
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join(" ".join(tknzr.tokenize(x)).split())),
        "label": df.label
    })


def clean_punctuation_2(df):
    """
    remove all punctuation and special characters
    """
    
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '/', '[', ']', "'", '@', '"', '\\', '!', '...', '..', '*', '<---'}
    
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([ "" if i in special_characters else i
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })


def remove_urls(df):
    """
    Remove all http https urls
    """
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([el if re.search("http.*", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })

    
def remove_ats(df):
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([el if re.search("^@", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })


def remove_ampersand(df):
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([el if re.search("^&", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })


def clean_empty_sentences(df):
    """
    Substitutes all the empty sentences with a customized sequence of characters
    """
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: re.sub("^ *$", "-", x)),
        "label": df.label
    })


def clean_with_vocabulary_of_model(df, vocabulary):
    """
    Given a model, it returns all the words in every sentence which are present in the vocabulary of the model
    :param df: the dataframe, with sentence and label column
    :param vocabulary: the vocabulary of the words known to the model.
    """
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([el if el in vocabulary else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })
