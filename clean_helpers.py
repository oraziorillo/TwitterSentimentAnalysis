import re
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

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
    remove all <...> occurrences
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
    
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '[', ']', "'", '@', '"', '\\'}
    
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
         
        print(words)    
        new_df.append({
            'sentence': " ".join(words),
            'label': row['label']
        })
    
    return pd.DataFrame(new_df)
