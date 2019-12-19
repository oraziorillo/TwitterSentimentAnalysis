from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from clean_helpers import *
import re
import json
import pandas as pd
import numpy as np


print('\n\
       ##############################\n\
       # Data cleaning              #\n\
       ##############################\n')

# Specify here what cleaning functions you want to use
cleaning_actions = ['clean_new_line', 'clean_tags', 'clean_punctuation', \
                    'remove_numbers']

clean = {
    "clean_new_line": clean_new_line,
    "lowercase": lowercase,
    "lemmatize": lemmatize,
    "remove_stopwords": remove_stopwords,
    "clean_punctuation": clean_punctuation,
    "clean_tags" : clean_tags,
    "remove_numbers": remove_numbers,
}

input_file_pos_full = 'Data/train_pos_full.txt'
input_file_neg_full = 'Data/train_neg_full.txt'
    
input_file_test = 'Data/test_data.txt'

# Create a dataframe containing all the sentences in train_pos_full.txt and train_neg_full.txt labeled
pos_sentences_full = []
with open(input_file_pos_full, 'r') as f:
    for sentence in f:
        pos_sentences_full.append(sentence)
        
neg_sentences_full = []
with open(input_file_neg_full, 'r') as f:
    for sentence in f:
        neg_sentences_full.append(sentence)
        
pos_data_full = pd.DataFrame(pos_sentences_full, columns=['sentence'])
pos_data_full['label'] = 1
neg_data_full = pd.DataFrame(neg_sentences_full, columns=['sentence'])
neg_data_full['label'] = 0

data = pd.concat([pos_data_full, neg_data_full])

# Create a dataframe containing all the sentences in test_data.txt labeled with a fake label
test_sentences = []
with open(input_file_test, 'r') as f:
    for sentence in f:
        test_sentences.append(sentence)

test_data = pd.DataFrame(test_sentences, columns=['sentence'])
test_data['label'] = -1

# Drop the indexes at the beginning of the sentences
test_data.sentence.apply(lambda x: re.sub("^[0-9]+,", "", x))

# Clean all the sentences in the dataframes
for c in cleaning_actions:
    print('\n>>>', c, 'on training set')
    data = clean[c](data)
    print('\n>>>', c, 'on test set')
    test_data = clean[c](test_data)
    print()
    
# Write the results on different csv files
print('\n>>> Saving clean training and test sets as csv files...')
data.to_csv('Data/clean_full_data.csv', index=False)
test_data.to_csv('Data/clean_test_data.csv', index=False)
print('Done')


print('\n\n\
       ##############################\n\
       # Data padding               #\n\
       ##############################\n')

# Create unique string containing all the sentences of the training dataset
all_text_clean = ' '.join(list(data['sentence']))

# Create a list of words (data is already tokenized)
words = all_text_clean.split()
# Count all the words
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}
vocabulary_lenght = len(vocab_to_int)

encoded_sentences = []
encoded_test_sentences = []

# Encode sentences by substituting each word with the respective encoding (according to the built vocabulary)

print('\n>>> Encoding training set')
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    encoded_sentence = [vocab_to_int[w] for w in row['sentence'].split()]
    encoded_sentences.append(encoded_sentence)
    
print('\n>>> Encoding test set')
for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    encoded_test_sentence = []
    for w in row['sentence'].split():
        try:
            encoded_test_sentence.append(vocab_to_int[w])
        except:
            # In case a word is not present in the vocabulary, put a neutral encoding 
            # (such as 0) in the corresponding encoded sentence
            encoded_test_sentence.append(0)

    encoded_test_sentences.append(encoded_test_sentence)

sentences_lenghts = [len(x) for x in encoded_sentences]
test_sentences_lenghts = [len(x) for x in encoded_test_sentences]
max_sentence_lenght = int(np.max(sentences_lenghts))

# Pad sequences with zeroes in front until the lenght is equal to max_sentence_lenght
print('\n>>> Padding sequences...')
X = pad_sequences(encoded_sentences, maxlen=max_sentence_lenght)
X_submission = pad_sequences(encoded_test_sentences, maxlen=max_sentence_lenght)
print('Done')

# Save parameters for lstm
print('\n>>> Saving configuration parameters...')
config = { 'vocabulary_len':len(vocab_to_int), 'max_sentence_lenght':max_sentence_lenght }
with open('config.json', 'w') as fp:
    json.dump(config, fp)
print('Done')

# Save the padded training and test sets on csv files
print('\n>>> Saving padded training and test sets as csv files...')
pd.DataFrame(X).to_csv('Data/pad_sentences.csv', index=False)
pd.DataFrame(X_submission).to_csv('Data/test_pad_sentences.csv', index=False)
print('Done\n')