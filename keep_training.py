import numpy as np
import pandas as pd
import sys
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re

# Load json model
json_file = open("model{}.json".format(sys.argv[1]), 'r') 
# Create model from json
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights into new model
model.load_weights("model{}.h5".format(sys.argv[1]))

print(model.summary())

iterations = 1
epochs = 5

embed_dim = 256
lstm_out = 196
batch_size = 256
test_size = 0.05
validation_size = 1000
vocabulary_lenght = 558852
max_sentence_lenght = 54

data = pd.read_csv('Data/clean_full_data.csv')
X = pd.read_csv('Data/pad_sentences.csv').values

model = Sequential()
model.add(Embedding(vocabulary_lenght, embed_dim, mask_zero=True, input_length=max_sentence_lenght))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

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
