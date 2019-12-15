import re
import pandas as pd

def clean_tags(df):
    """
    remove all <...> occurrences
    """
    tag = "^<.*>$"
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x:  " ".join(" ".join([ w if w != "" and re.search(tag, w) is None else "" for w in x.split(" ")]).split())),
        'label': df.label
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
    
    special_characters = {'.', ',', '<', '>', '(', ')', ':', ';', '/', '[', ']', "'", '@', '"', '\\'}
    
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([ "" if i in special_characters else i
                                                         for i in x.split(" ")]).split()) ),
        'label': df['label']
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

    
def remove_numbers(df):
    """
    removes numbers, they are useless
    """
    return pd.DataFrame({
        'sentence': df.sentence.apply(lambda x: " ".join(" ".join([el if re.search("^([-\.,/_\(\):#x]?[0-9]+[-\.,/_\(\):x]?)+$", el) == None else "" for el in x.split()]).split()) ),  
        'label': df['label']
    })
