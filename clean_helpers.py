import re
import pandas as pd
from textblob import TextBlob, Word
import nltk
from nltk.corpus import stopwords

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
    
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '/', '[', ']', "'", '@', '"', '\\', '-'}
    
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
    print(len(stopw))

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
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join( j if not j.isdigit() else "" for j in [ i.replace(".", "", 1)
                                                         for i in x.split(" ")]).split()) ),
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
