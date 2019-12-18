from clean_helpers import *
import pandas as pd

# Specify here what cleaning functions you want to use
cleaning_actions = ['clean_new_line', 'clean_tags', 'clean_punctuation', \
                    'remove_numbers']

clean = {
    "clean_new_line": clean_new_line,
    "lowercase": lowercase,
    "lemmatize": lemmatize,
    "remove_stopwords": remove_stopwords,
    "translate": perform_translation,
    "clean_punctuation": clean_punctuation,
    "clean_tags" : clean_tags,
    "remove_numbers": remove_numbers,
}

input_file_pos_full = 'Data/train_pos_full.txt'
input_file_neg_full = 'Data/train_neg_full.txt'

input_file_pos = 'Data/train_pos.txt'
input_file_neg = 'Data/train_neg.txt'
    
input_file_test = 'Data/test_data.txt'


# Create a dataframe containing all the sentences in train_pos.txt and train_neg.txt labeled
pos_sentences = []
with open(input_file_pos, 'r') as f:
    for sentence in f:
        pos_sentences.append(sentence)
        
neg_sentences = []
with open(input_file_neg, 'r') as f:
    for sentence in f:
        neg_sentences.append(sentence)
        
pos_data = pd.DataFrame(pos_sentences, columns=['sentence'])
pos_data['label'] = 1
neg_data = pd.DataFrame(neg_sentences, columns=['sentence'])
neg_data['label'] = 0

data = pd.concat([pos_data, neg_data])


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

full_data = pd.concat([pos_data_full, neg_data_full])


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
    data = clean[c](data)
    full_data = clean[c](full_data)
    test_data = clean[c](test_data)

    
# Write the results on different csv files    
data.to_csv('Data/clean_data.csv', index=False)
full_data.to_csv('Data/clean_full_data.csv', index=False)
test_data.to_csv('Data/clean_test_data.csv', index=False)