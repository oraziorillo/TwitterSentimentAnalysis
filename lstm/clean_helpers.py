from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
from tqdm import tqdm
import pandas as pd
import nltk
import spacy  # For lemmatization.
import re
from nltk.tokenize import TweetTokenizer  # This is used to tokenize the second dataset.


def clean_tags(df):
    """
    remove all <...> occurrences
    """
    tqdm.pandas()
    tag = "^<.*>$"
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x:  " ".join(" ".join([ w if w != "" and re.search(tag, w) is None else "" for w in x.split(" ")]).split())),
        'label': df.label
    })
    

def lowercase(df):  
    tqdm.pandas()
    return pd.DataFrame({
        "sentence": df.sentence.progress_apply(lambda x: " ".join([i.lower() for i in x.split(" ")])),
        "label": df.label
    })


def clean_new_line(df):
    """
    remove all \n
    Careful here: are we sure that we have no \n and whitespace together?
    We replace all occurrences of double whitespace with a single space!
    """
    tqdm.pandas()
    return pd.DataFrame({
        "sentence": df.sentence.progress_apply(lambda x: " ".join((" ".join(x.split("\n")[:])).split())),  # We replace all occurrences of 2 and 3 whitespace with 1 whitespace!
        "label": df.label    
    })


def clean_punctuation(df):
    """
    remove all punctuation and special characters
    """
    
    tqdm.pandas()
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '/', '[', ']', "'", '@', '"', '\\', '-', '&', '*', '+', '#', '|', '^', '~'}
    
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([ "" if i in special_characters else i
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })



def clean_punctuation_2(df):
    """
    remove all punctuation and special characters
    """
    
    tqdm.pandas()
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '/', '[', ']', "'", '@', '"', '\\', '!', '...', '..', '*', '<---'}
    
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([ "" if i in special_characters else i
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })


def remove_saxon_genitive(df):
    """
    remove all 's occurrences
    """
    
    tqdm.pandas()
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([ i.replace("'s", "")
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })
    

def remove_stopwords(df):

    nltk.download('stopwords')

    stopw = set(stopwords.words('english'))
    print("The number of scipy stopwords is {}".format(len(stopw)))

    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([ "" if i in stopw else i
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
    })


def remove_numbers(df):
    """
    removes numbers, they are useless
    """
    tqdm.pandas()
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([el if re.search("^([-\.,/_\(\):x]?[0-9]+[-\.,/_\(\):x]?)+$", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })



def lemmatize(df):
    """
    Lemmatize tokens
    """

    nltk.download('wordnet') 
    tqdm.pandas()

    return pd.DataFrame({
        "sentence": df.sentence.progress_apply(lambda x: " ".join([ Word(w).lemmatize() for w in x.split(" ")])),
        "label": df.label
    })


def lemmatize_spacy(df):
    """
    Do lemmatization with spacy library, looks like it is working better than
    TextBlob. Only thing, must be very careful with hashtags (they get split.)
    Probably better doing some more eye check controls to see if everything is
    OK. Other strange fact, it substitutes every pronoun as -PRON-
    """
    tqdm.pandas()
    nlp = spacy.load('en', disable=['parser', 'ner'])   # Load the spacy nlp module, in english.
    return pd.DataFrame({
        "sentence": df.sentence.progress_apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]).replace("# ", "#")),
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
    Cleans duplicate characters wherever they are repeated more than twice. 
    """    
    
    tqdm.pandas()
    return pd.DataFrame({
        "sentence": df.sentence.progress_apply(lambda x: " ".join([reduce_lengthening(w) for w in x.split()]) ),
        "label": df.label
    })


def tokenize_tweets(df):
    tknzr = TweetTokenizer()
    tqdm.pandas()
    return pd.DataFrame({
        "sentence": df.sentence.progress_apply(lambda x: " ".join(" ".join(tknzr.tokenize(x)).split())),
        "label": df.label
    })


def remove_urls(df):
    """
    Remove all http.* urls
    """
    tqdm.pandas()
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([el if re.search("http.*", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })

    
def remove_ats(df):
    tqdm.pandas()
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([el if re.search("^@", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })


def remove_ampersand(df):
    tqdm.pandas()
    return pd.DataFrame({
        'sentence': df.sentence.progress_apply(lambda x: " ".join(" ".join([el if re.search("^&", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })


def clean_empty_sentences(df):
    """
    Substitutes all the empty sentences with a customized sequence of characters
    """
    tqdm.pandas()
    return pd.DataFrame({
        "sentence": df.sentence.progress_apply(lambda x: re.sub("^ *$", "-", x)),
        "label": df.label
    })


def clean_duplicate_sentences(df):
    """
    Clean all the duplicate sentences, leaving just one copy
    """
    return df.drop_duplicates(subset ="sentence")
