import pickle
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Builds a split into train and test data for local tests.')

parser.add_argument('--input_pickle',
                    required=True,
                    help='input file (pickle for a dataframe with sentence and label columns)')
parser.add_argument('--split',
                    required=True,
                    help='fraction of the dataframe that will be used for training, and testing (1-split)')

parser.add_argument('--output_train',
                    required=True,
                    help='file name where the train dataframe will be put (pickle format)')

parser.add_argument('--output_test',
                    required=True,
                    help='file name where the test dataframe will be put (pickle format)')

args = parser.parse_args()

# Get the dataframe
df = pd.read_pickle(args.input_pickle)

train_test_split = float(args.split)

# Create a random permutation
permut = np.random.permutation(df.shape[0])

# Select the first part of the permutation as the training set, and the remaining as the test set.
train = df.iloc[permut[: int(df.shape[0]*train_test_split)]]
test = df.iloc[permut[int(df.shape[0]*train_test_split): ]]

print("the size of train is {}".format(len(train)))
print("the size of test is {}".format(len(test)))

# Store the dataframes to file.
train.to_pickle(args.output_train)
test.to_pickle(args.output_test)

