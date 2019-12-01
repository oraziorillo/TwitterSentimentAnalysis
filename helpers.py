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

