from helpers import help_generate_predictions
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import pandas as pd
import numpy as np
import sys
import re


if len(sys.argv) != 2:
    print("\n\tWrong number of arguments")
    help_generate_predictions()
    sys.exit(0)

try:
    model_name = sys.argv[1]
except:
    print("\n\tNot valid model name")
    help_generate_predictions()
    sys.exit(0)

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

embedded_test_data = pd.read_csv('Data/bert_sentence_embeddings_test.csv')
X_test = embedded_test_data.values.tolist()
                   
predictions = []
                   
for i in range(len(X_test)):
    prediction = model.predict(X_test[i].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
    predictions.append(1 if prediction[0] >= prediction[1] else -1)

results = pd.DataFrame({
    "Id": np.arange(1, X_test.shape[0] + 1),
    "Prediction": predictions
})

results.to_csv('Submission.csv', index=False)