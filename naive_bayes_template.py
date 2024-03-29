import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

from tqdm import tqdm

from naive_bayes import naive_bayes_model, naive_bayes_predict
from clean_helpers import *
from data_handling import build_sentences

from helpers import write

sys.path.append('../')

import argparse

parser = argparse.ArgumentParser(description='It performs naive Bayes, when the test_locally flag' + 
    "is set to zero, it will plot the accuracy graph for increasing length of ngrams. Otherwise, "+
    "It will create the submission file.")

parser.add_argument('--take_full',
                    required=True,
                    help='take full dataset of tweets (either 0 or 1)')
parser.add_argument('--test_locally',
                    required=True,
                    help='Either 0 or 1, if 1 it will create the submission csv file.')
parser.add_argument('--output_file',
                    required= False,
                    help='The output file for the submission, OPTIONAL')



args = parser.parse_args()


if __name__ == "__main__":
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
        # Generate the plot of accuracies for increasing maximum length of the n-grams
        accuracies = []
        n=11

        for i in tqdm(range(1,n)):
            model = naive_bayes_model(train['sentence'], train['label'], i)
            prediction = naive_bayes_predict(model, test['sentence'])
            accuracy = np.mean(np.where(prediction==test['label'],1,0))
            accuracies.append(accuracy)
        plt.plot(np.arange(1,n), accuracies)
        plt.title('accuracy for naive Bayes for increasing maximum length of n-grams')
        plt.xlabel('n')
        plt.ylabel('accuracy')
        plt.show()
    else:
        # Create a submission file
        model = naive_bayes_model(train['sentence'], train['label'], 1)
        prediction = naive_bayes_predict(model, test['sentence'])
        results = pd.DataFrame({'Id':range(1, len(prediction)+1), 'Prediction':prediction})
        results.to_csv('Submission.csv', index=False)
        output_file = args.output_file
        write(prediction, output_file)
