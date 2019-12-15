import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from collections import Counter
from clean_helpers import *
import re
import sys

sys.path.append('../')

embed_dim = 128
lstm_out = 196 
batch_size = 128
epochs = 5
test_size = 0.10
validation_size = 1000
iterations = 4

take_full = True
test_locally = True

# Specify here what cleaning functions you want to use
cleaning_actions = ['clean_new_line', 'clean_tags', 'clean_punctuation', \
                    'remove_numbers']

clean = {
    "clean_new_line": clean_new_line,
    "clean_punctuation": clean_punctuation,
    "clean_tags" : clean_tags,
    "remove_numbers": remove_numbers,
}

if take_full:
    input_file_pos = 'Data/train_pos_full.txt'
    input_file_neg = 'Data/train_neg_full.txt'
else:
    input_file_pos = 'Data/train_pos.txt'
    input_file_neg = 'Data/train_neg.txt'
    
input_file_test = 'Data/test_data.txt'

pos_sentences = []
with open(input_file_pos, 'r') as f:
    for sentence in f:
        pos_sentences.append(sentence)
        
neg_sentences = []
with open(input_file_neg, 'r') as f:
    for sentence in f:
        neg_sentences.append(sentence)

test_sentences = []
with open(input_file_test, 'r') as f:
    for sentence in f:
        test_sentences.append(sentence)

pos_data = pd.DataFrame(pos_sentences, columns=['sentence'])
pos_data['label'] = 1
neg_data = pd.DataFrame(neg_sentences, columns=['sentence'])
neg_data['label'] = -1
test_data = pd.DataFrame(test_sentences, columns=['sentence'])
test_data['label'] = 0

data = pd.concat([pos_data, neg_data])

for c in cleaning_actions:
    data = clean[c](data)
    test_data = clean[c](test_data)

all_text_clean = ' '.join(list(data['sentence']))

# Create a list of words
words = all_text_clean.split()
# Count all the words using Counter Method
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}

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

model = Sequential()
model.add(Embedding(len(vocab_to_int), embed_dim, mask_zero=True, input_length=max_sentence_lenght))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())

for i in range(iterations):
    Y = pd.get_dummies(data['label']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_size, random_state = 42)

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model{}.json".format(i), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model{}.h5".format(i))
    print("Saved model{} to disk".format(i))
