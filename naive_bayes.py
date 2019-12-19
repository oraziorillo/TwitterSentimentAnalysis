from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def naive_bayes_model(training_data, training_labels, n):
    """
    Create a naive Bayes classifier pipeline with maximum maximum length of n-gram given.

    :param pandas.core.series.Series training_data: The data the naive Bayes classifier should be trained on
    :param pandas.core.series.Series training_labels: The labels corresponding to the given training data
    :param int n: The maximum value of n in the sense of n-grams
    """
    classifier = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,n))), #add arguments to CountVectorizer for more efficiency
        ('tfidf', TfidfTransformer(sublinear_tf=False)),
        ('clf', MultinomialNB()),
    ])
    classifier.fit(training_data, training_labels)
    return classifier


def naive_bayes_predict(classifier, test_data):
    """
    Predict the labels of the given test data with the given classifier pipeline.
    
    :param sklearn.pipeline.Pipeline classifier: The classifier pipeline with which the labels of the given test data should be predicted
    :param pandas.core.series.Series test_data: The data the given classifier pipeline should predict the labels of
    """
    return classifier.predict(test_data)