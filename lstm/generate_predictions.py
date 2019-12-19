from keras.models import model_from_json
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# Parsing arguments
parser = argparse.ArgumentParser(description='Train lstm-NN model.')
parser.add_argument('-m, --model', dest='model_name', required=True,
                    help='name of the model to load ad input (without extension)')
parser.add_argument('-o, --output', dest='output_name', default='submission',
                    help='name of output csv file')
args = parser.parse_args()

model_name = vars(args)['model_name']
output_name = vars(args)['output_name']

# Read padded data
X = pd.read_csv('Data/pad_sentences.csv').values
X_test = pd.read_csv('Data/test_pad_sentences.csv').values

# Load json model
print('\n>>> Loading {}...'.format(model_name))
try: 
    json_file = open("trained_models/{}.json".format(model_name), 'r')
except:
    print('No models with that name in trained_models')
        
# Create model from json
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights into new model
model.load_weights("trained_models/{}.h5".format(model_name))
print('Done\n')
print(model.summary())
                 
print('\n>>> Generating predictions...')
predictions = []
for x in tqdm(range(len(X_test))):
    # Generate predictions
    prediction = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
    predictions.append(-1 if prediction[0] >= prediction[1] else 1)

print('\n>>> Saving results in results as submission.csv')
results = pd.DataFrame({
    "Id": np.arange(1, X_test.shape[0] + 1),
    "Prediction": predictions
})
print('Done')

results.to_csv('{}.csv'.format(output_name, index=False))
