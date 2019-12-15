from nltk import ngrams
import pandas as pd

def create_labelled_file(name_file, train):
    """
    Create the input file for fasttext
    it is made using the __label__ prefix for 1, -1 (labels for :), :( )
    and the sentence itself
    """
    with open(name_file, 'w') as f:
        for index, row in train.iterrows():
            f.write("__label__{} {}\n".format(row['label'], row['sentence']))
            
    # A bit useless, but may be clearer
    return name_file


def count_unique_words(df):
    """
    counts the number of unique words in the vocabulary
    """
    df_list = df.sentence.apply(lambda x: x.split(" "))
    df_list_exploded = df_list.explode()
    return len(df_list_exploded.unique())


def count_unique_ngrams(df, n):
    """
    Count the number of ngrams
    """
    return len(df.sentence.apply(lambda x: [x for x in ngrams(x.split(), n)]).explode().unique())


# -------------------------------------------------------------------------------------------------------------------
# Bert stuff
# -------------------------------------------------------------------------------------------------------------------

def help_bert_sentence_embeddings():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 bert_sentence_embeddings.py <embeddings_lenght> <sentences_to_embed.csv>")
    

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


# -------------------------------------------------------------------------------------------------------------------
# LSTM stuff
# -------------------------------------------------------------------------------------------------------------------

def help_train_lstm():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 help_train_lstm.py <number_of_epochs>")
    
def help_keep_training_lstm():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 help_keep_training_lstm.py <model_name> <number_of_epochs>")