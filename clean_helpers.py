import re
import pandas as pd
from textblob import TextBlob, Word
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import spacy
from spacy_langdetect import LanguageDetector

def clean_tags(df):
    """
    remove all <...> occurrences
    """
    new_df = []
    for index, row in df.iterrows():
        list_of_words = row['sentence'].split(" ")
        new_list = []
        for w in list_of_words:
            tag = "^<.*>$"
            if re.search(tag, w) is None:
                if w != '':
                    new_list.append(w)
        new_df.append({
            'sentence': " ".join(new_list),
            'label': row['label']
        })
    
    return pd.DataFrame(new_df)
    

def lowercase(df):
    """
    lowercase things
    """
    new_df = []
    for index, row in df.iterrows():
        list_of_words = row['sentence'].split(" ")
        new_list = []
        for w in list_of_words:
            new_list.append(w.lower())
        new_df.append({
            'sentence': " ".join(new_list),
            'label': row['label']
        })
    
    return pd.DataFrame(new_df)  

def clean_new_line(df):
    """
    remove all \n
    """
    new_df = []
    for index, row in df.iterrows():
        new_string = "".join(row['sentence'].split("\n")[0:-1])
        new_df.append({
            'sentence': new_string,
            'label': row['label']
        })
    
    return pd.DataFrame(new_df)  

def clean_punctuation(df):
    """
    remove all punctuation and special characters
    """
    
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '/', '[', ']', "'", '@', '"', '\\'}
    
    new_df = []
    for index, row in df.iterrows():
        words = []
        new_string = row['sentence'].split(" ")
        
        for w in new_string:
            if w not in special_characters:
                words.append(w)
         
        new_df.append({
            'sentence': " ".join(words),
            'label': row['label']
        })
    
    return pd.DataFrame(new_df)


def remove_stopwords(df):
    stopw = set(stopwords.words('english'))
    new_df = []
    for index, row in df.iterrows():
        words = []
        new_string = row['sentence'].split(" ")
        
        for w in new_string:
            if w not in stopw:
                words.append(w)
                
        new_df.append({
            'sentence': " ".join(words),
            'label': row['label']
        })
    
    return pd.DataFrame(new_df)



def perform_translation(df):
    """
    Up to now it just computes how many languages are present, and how often
    """
    nlp = spacy.load('en')  # load english library (probably not necessary!)
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)  # Add the tranlator to the pipeline
    languages_detected = []
    counter_of_empty_sentences = 0
    for index, row in tqdm(df.iterrows()):
        sentence = row['sentence']
        doc = nlp(sentence)
        languages_detected.append(doc._.language)
        
    language_df = pd.DataFrame(languages_detected)
    print(language_df.head())
    
    print(language_df.groupby("language").count())
    
    # language_df.hist(column=['language'])
    # In the meantime don't do anything
    return df

    
def remove_numbers(df):
    """
    removes numbers, they are useless
    """
    new_df = []
    for index, row in df.iterrows():
        words = []
        for w in row['sentence'].split(" "):
            if not w.replace('.','', 1).isdigit():
                words.append(w)
    
        new_df.append({
            'sentence': " ".join(words),
            'label': row['label']
        })
        
    return pd.DataFrame(new_df)


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
    new_df = []
    for index, row in df.iterrows():
        words = []
        for w in row['sentence'].split(" "):
            w = Word(w)
            words.append(w.lemmatize())

        new_df.append({
            'sentence': " ".join(words),
            'label': row['label']
        })
    return pd.DataFrame(new_df)
        