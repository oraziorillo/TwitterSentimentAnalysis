import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from clean_helpers import *
from collections import Counter
import sys
import os

os.chdir('..')

data = pd.read_csv('Data/clean_full_data.csv')
test_data = pd.read_csv('Data/clean_test_data.csv')

all_text_clean = ' '.join(list(data['sentence']))

# Create a list of words
words = all_text_clean.split()
# Count all the words using Counter Method
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}

print('len(vocab_to_int) =', len(vocab_to_int))

encoded_sentences = []
encoded_test_sentences = []

for index, row in data.iterrows():
    encoded_sentence = [vocab_to_int[w] for w in row['sentence'].split()]
    encoded_sentences.append(encoded_sentence)

for index, row in test_data.iterrows():
    encoded_test_sentence = []
    for w in row['sentence'].split():
        try:
            encoded_test_sentence.append(vocab_to_int[w])
        except:
            encoded_test_sentence.append(0)

    encoded_test_sentences.append(encoded_test_sentence)

sentences_lenghts = [len(x) for x in encoded_sentences]
test_sentences_lenghts = [len(x) for x in encoded_test_sentences]
max_sentence_lenght = np.max([np.max(sentences_lenghts), np.max(test_sentences_lenghts)])

X = pad_sequences(encoded_sentences, maxlen=max_sentence_lenght)
X_submission = pad_sequences(encoded_test_sentences, maxlen=max_sentence_lenght)

pd.DataFrame(X).to_csv('Data/pad_sentences.csv', index=False)
pd.DataFrame(X_submission).to_csv('Data/test_pad_sentences.csv', index=False)
