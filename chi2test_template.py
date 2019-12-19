import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter

from data_handling import build_sentences

from clean_helpers import *

sys.path.append('../')


def chi2_test_naive_bayes(n, k):
    """
    Create a naive Bayes pipeline including a SelectKBest transformation.
    
    :param int n: The maximum value of n in the sense of n-grams
    :param int k: The number of best scoring features on the chi2-test that should be selected
    """
    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,n))),
        ('chi2', SelectKBest(chi2, k=k)),
        ('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=True)),
        ('clf', MultinomialNB()),
    ])    
    return text_clf


def chi2_test_svm(k):
    """
    Create a SVM pipeline including a SelectKBest transformation.
    
    :param int k: The number of best scoring features on the chi2-test that should be selected
    """
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('chi2', SelectKBest(chi2, k=k)),
        ('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=True)),
        ('clf', LinearSVC()),
    ])    
    return text_clf


def chi2_test(training_data, training_labels, testing_data, testing_labels):
    """
    Perform the chi2-test on the given data for the given maximimum length of n-grams.
    
    :param pandas.core.series.Series data: The training data on which the chi2-test is performed
    :param pandas.core.series.Series data: The labels of the training data on which the chi2-test is performed
    :param pandas.core.series.Series data: The testing data on which the chi2-test is performed
    :param pandas.core.series.Series data: The labels of the testing data on which the chi2-test is performed
    """
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(training_data)
    lexicon = count_vect.get_feature_names()

    min_zoom = 6000
    accuracies = []

    # log distribuition from 1 to n_features
    k_values = set(np.geomspace(1,len(lexicon), num=20, dtype=np.int))
    k_values = sorted(k_values.union(set(np.geomspace(min_zoom, len(lexicon), num=20, dtype=np.int))))

    for k_value in tqdm(k_values):
        if params['model']=='naive_bayes':
            n = params['n']
            text_clf = chi2_test_naive_bayes(n, k_value)
        else:
            text_clf = chi2_test_svm(k_value)  
        model = text_clf.fit(training_data, training_labels)
        prediction = text_clf.predict(testing_data)
        accuracy = np.mean(prediction == testing_labels)
        accuracies.append(accuracy)
    
    k_values = np.array(k_values)
    accuracies = np.array(accuracies)

    print(params['model']+", n="+str(params['n'])+': highest accuracy of {} achieved with top {} features used'.format(accuracies.max(), k_values[accuracies.argmax()]))
    
    plt.figure(figsize=(10,8))
    plt.plot(k_values, accuracies, marker='.')
    plt.xscale('log')
    plt.xlabel('k features used')
    plt.minorticks_off()
    plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
    plt.ylabel('Accuracy')
    using_all = accuracies[-1]
    plt.plot(k_values,using_all*np.ones_like(k_values), color='orange', label='Using all features')
    plt.legend()
    plt.show()


if __name__=="__main__":
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
    input_file_pos = 'Data/train_pos_full.txt'
  
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
    shuffled = df.sample(frac=1)
    train = shuffled[:int(len(df)*0.7)]
    test = shuffled[int(len(df)*0.7)+1:]
    print(train.shape, test.shape)

    print(df.count())
    print(train[train.label == -1].count())
    print(test[test.label == 1].count())
        
        
    # Defining the parameters of the chi2-test
    params = {
        'model': 'naive_bayes',
        'n': 1 # the maximum length in terms of n-grams
    }
    
    
    # Performing the chi2-test for several cases
    chi2_test(train['sentence'], train['label'], test['sentence'], test['label'])
    
    params['n'] = 2
    chi2_test(train['sentence'], train['label'], test['sentence'], test['label'])
    
    params['n'] = 1
    params['model'] = 'svm'
    chi2_test(train['sentence'], train['label'], test['sentence'], test['label'])