from helpers import help_keep_training_lstm
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import pandas as pd
import sys
import re


if len(sys.argv) != 3:
    print("\n\tWrong number of arguments")
    help_keep_training_lstm()
    sys.exit(0)

try:
    model_name = sys.argv[1]
    model_number = int(re.compile('[0-9]+$').findall(model_name)[0])
except:
    print("\n\tNot valid model name")
    help_keep_training_lstm()
    sys.exit(0)

try:
    epochs = int(sys.argv[2])
except:
    print("\n\tNot valid number of epochs")
    help_keep_training_lstm()
    sys.exit(0)


embed_dim = 128
lstm_out = 196 
batch_size = 128
test_size = 0.10
validation_size = 1000


# Load json model
json_file = open("models/{}.json".format(model_name), 'r')

# Create model from json
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights into new model
model.load_weights("models/{}.h5".format(sys.argv[1])
print("Loaded model from disk")
print(model.summary())
                   
# Compile model
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

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
with open("models/bert_lstm_{}.json".format(model_number + 1), "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights("models/bert_lstm_{}.h5".format(model_number + 1))
print("Saved model bert_lstm_{} to disk".format(model_number + 1))

