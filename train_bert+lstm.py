from helpers import help_train_lstm
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import re


if len(sys.argv) != 2:
    print("\n\tWrong number of arguments")
    help_train_lstm()
    sys.exit(0)

try:
    epochs = int(sys.argv[2])
except:
    print("\n\tNot valid number of epochs")
    help_train_lstm()
    sys.exit(0)


embed_dim = 128
lstm_out = 196 
batch_size = 128
test_size = 0.10
validation_size = 1000

# Initialize lstm model
model = Sequential()
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

# Import data
X_df = pd.read_csv("Data/bert_sentence_embeddings_full.csv")
X = X_df.values.tolist()

data = pd.read_csv("Data/clean_full_data.csv")
Y = pd.get_dummies(data['label']).values
                   
# Split into training an test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_size, random_state = 42)

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

# Serialize model to JSON
model_json = model.to_json()
with open("models/bert_lstm_0.json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights("models/bert_lstm_0.h5")
print("Saved model bert_lstm_0 to disk")