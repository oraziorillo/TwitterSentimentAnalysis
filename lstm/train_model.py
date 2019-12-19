import pandas as pd
import argparse
import json

# Parsing arguments
parser = argparse.ArgumentParser(description='Train lstm-NN model.')
parser.add_argument('-i, --iterations', dest='iterations', type=int, required=True,
                    help='number of trainings performed by the model')
parser.add_argument('-ep, --epochs', dest='epochs',  type=int, required=True,
                    help='number of epochs performed per training')
parser.add_argument('-es, --embedding', dest='embed_dim',  type=int, default=256,
                    help='size of the embedding vectors')
parser.add_argument('-bs, --batch', type=int, dest='batch_size',  default=256,
                    help='size of the batch')
parser.add_argument('-ts, --test', type=float, dest='test_size',  default=0.05,
                    help='proportion between the test set and the training set during the splitting of the dataset')
parser.add_argument('-l, --loss', dest='loss',  default='binary_crossentropy',
                    help='loss function used by the lstm')
parser.add_argument('-a, --activation', dest='activation',  default='softmax',
                    help='activation function used by the NN to produce the output')
parser.add_argument('-n, --name', dest='model_name', default='model',
                    help='name to give to the saved model (without extension)')

args = parser.parse_args()

iterations = vars(args)['iterations']
epochs = vars(args)['epochs']
activation = vars(args)['activation']
loss = vars(args)['loss']
embed_dim = vars(args)['embed_dim']
batch_size = vars(args)['batch_size']
test_size = vars(args)['test_size']
model_name = vars(args)['model_name']

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

# Take parameters from config file
with open('config.json') as json_file:
    config = json.load(json_file)

max_sentence_lenght = config['max_sentence_lenght']
vocabulary_lenght = config['vocabulary_len']

# Load data
print('\n>>> Loading data...')
data = pd.read_csv('Data/clean_full_data.csv')
X = pd.read_csv('Data/pad_sentences.csv').values
print('Done')

print('\n>>> Building model...\n')

model = Sequential()
model.add(Embedding(vocabulary_lenght, embed_dim, mask_zero=True, input_length=max_sentence_lenght))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation=activation))
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

print(model.summary())
print('\n>>> Training...\n')

for i in range(iterations):
    Y = pd.get_dummies(data['label']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_size, random_state = 42)

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate model using score and accuracy as metrics
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print()
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

    # Serialize model to JSON
    model_json = model.to_json()
    with open("trained_models/{}{}.json".format(model_name,i), "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("trained_models/{}{}.h5".format(model_name,i))
    print("\nSaved {}{} in trained_models\n".format(model_name,i))
