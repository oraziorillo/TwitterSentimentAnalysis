import re
import pandas as pd
from textblob import TextBlob, Word
import nltk
import gensim
from nltk.corpus import stopwords
from pattern.en import spelling

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
        "sentence": df.sentence.apply(lambda x: " ".join((" ".join(x.split("\n")[0:-1])).split())),  # We replace all occurrences of 2 and 3 whitespace with 1 whitespace!
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


def perform_translation(df):
    """
    perform translation of sentences. 
    I don't know yet if it is better to perform it on a sentence level
    or on a word level.
    
    DOESN'T WORK!! Google api doesn't accept too many requests.
    """
    languages_detected = []
    counter_of_empty_sentences = 0
    for index, row in df.iterrows():
        sentence = row['sentence']
        if len(sentence) > 3:
            b = TextBlob(sentence)
            if b.detect_language() != 'en':
                languages_detected.append(b.detect_language())
        else:
            counter_of_empty_sentences += 1 
            
    
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
    Lemmatize things:
    in order to work: 
    1) open an ipython3 shell on the correct virtualenv
        `ipython3`
    2) type: 
        `import nltk`
        `nltk.download('wordnet')`
    3) exit()
    """
    
    return pd.DataFrame({
        "sentence": df.sentence.apply(lambda x: " ".join([ Word(w).lemmatize() for w in x.split(" ")])),
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
    

