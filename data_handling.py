import pandas as pd
from textblob import TextBlob

def build_sentences(list_of_pos_sentences, list_of_neg_sentences):
    list_of_sentences = []
    for l in list_of_pos_sentences:
        list_of_sentences.append({
            "sentence": l,
            "label": 1
        })
    
    for l in list_of_neg_sentences:
        list_of_sentences.append({
            "sentence": l,
            "label": -1
        })
    df = pd.DataFrame(list_of_sentences)
    
    return df