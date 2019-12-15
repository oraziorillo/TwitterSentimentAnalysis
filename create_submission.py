import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from collections import Counter
from clean_helpers import *
import sys

input_file_pos = 'Data/train_pos_full.txt'
input_file_neg = 'Data/train_neg_full.txt'
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

# Drop the indexes at the beginning of the sentences
test_data.sentence.apply(lambda x: re.sub("^[0-9]+,", "", x))

data = pd.concat([pos_data, neg_data])

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

# Load json model
json_file = open("models/model{}.json".format(sys.argv[1]), 'r')

# Create model from json
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights into new model
model.load_weights("models/model{}.h5".format(sys.argv[1]))
print("Loaded model from disk")
print(model.summary())
                   
predictions = []
for x in range(len(X_submission)):
    
    prediction = model.predict(X_submission[x].reshape(1,X_submission.shape[1]),batch_size=1,verbose = 2)[0]
    predictions.append(1 if prediction[0] >= prediction[1] else -1)

results = pd.DataFrame({
    "Id": np.arange(1, X_submission.shape[0] + 1),
    "Prediction": predictions
})

results.to_csv('Submission.csv', index=False)