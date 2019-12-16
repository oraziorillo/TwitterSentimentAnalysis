import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import sys
import os

os.path.dirname('..')

X = pd.read_csv('Data/pad_sentences.csv').values
X_submission = pd.read_csv('Data/test_pad_sentences.csv').values

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
    predictions.append(-1 if prediction[0] >= prediction[1] else 1)

results = pd.DataFrame({
    "Id": np.arange(1, X_submission.shape[0] + 1),
    "Prediction": predictions
})

results.to_csv('Submission.csv', index=False)
