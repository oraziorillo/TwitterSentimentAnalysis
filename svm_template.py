import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

from tqdm import tqdm

from svm import svm_model, svm_predict
from clean_helpers import *
from data_handling import build_sentences

sys.path.append('../')

import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--take_full',
                    help='take full dataset of tweets')
parser.add_argument('--test_locally',
                    help='Input file with negative tweets')



args = parser.parse_args()

if _name=="main_":
    # Specify here whether or not you want to consider the full dataset, and whether or not you want to test locally
    take_full = True if args.take_full == '1' else False
    test_locally = True if args.test_locally == '1' else False

    
    # Specify here what cleaning functions you want to use
    cleaning_options = ['clean_new_line', 'clean_tags', 'lowercase', \
                        'clean_punctuation', 'remove_stopwords', 'remove_numbers', 'lemmatize']

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
    
    
    # Specifying the datasets
    input_file_pos = 'Data/train_pos.txt'
    if take_full:
        input_file_pos = 'Data/train_pos_full.txt'
  
    input_file_neg = 'Data/train_neg.txt'
    if take_full:
        input_file_neg = 'Data/train_neg_full.txt'
    
    list_of_pos_sentences = []
    with open(input_file_pos, 'r') as f:
        for line in f:
            list_of_pos_sentences.append(line)
 
    list_of_neg_sentences = []
    with open(input_file_neg, 'r') as f:
        for line in f:
            list_of_neg_sentences.append(line)
    
    
    # Building the sentences
    df = build_sentences(list_of_pos_sentences, list_of_neg_sentences)
    df.head()

    for clean_option in cleaning_options:
        df = clean[clean_option](df)
        print(clean_option)
        print(df[['sentence']][:100])
        print("################################\n\n")
        
        
    # Building the train and test sets
    if test_locally:
        shuffled = df.sample(frac=1)
        train = shuffled[:int(len(df)*0.7)]
        test = shuffled[int(len(df)*0.7)+1:]
        print(train.shape, test.shape)

        print(df.count())
        print(train[train.label == -1].count())
        print(test[test.label == 1].count())
    else:
        train = df
        test = []
        with open("Data/test_data.txt", 'r') as f:
            for l in f:
                id_ = l.split(",")[0]
                sentence = ",".join(l.split(",")[1:])
                test.append({
                    "label": int(id_),
                    "sentence": sentence
                })
        test = pd.DataFrame(test)
        test.head()
    
        for clean_option in cleaning_options:
            test = clean[clean_option](test)
            print(clean_option)
            print(test[['sentence']][:100])
            print("################################\n\n")
            

    if test_locally:
        # Compute the accuracy for the SVM
        model = svm_model(train['sentence'], train['label'])
        prediction = svm_predict(model, test['sentence'])
        accuracy = np.mean(np.where(prediction==test['label'],1,0))
        print("the SVM yields an accuracy of "+str(accuracy))
    else:
        # Create a submission file
        model = svm_model(train['sentence'], train['label'], 1)
        prediction = svm_predict(model, test['sentence'])
        results = pd.DataFrame({'Id':range(1, len(prediction)+1), 'Prediction':prediction})
        results.to_csv('Submission.csv', index=False)
        output_file = "output/output.csv"
        write(prediction, output_file)

