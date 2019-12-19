from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC


def svm_model(training_data, training_labels):
    """
    Create an SVM classifier pipeline.

    :param pandas.core.series.Series training_data: The data the SVM should be trained on
    :param pandas.core.series.Series training_labels: The labels corresponding to the given training data
    """
    classifier = Pipeline([
        ('vect', CountVectorizer()), #add arguments to CountVectorizer for more efficiency
        ('tfidf', TfidfTransformer(sublinear_tf=False)),
        ('clf', LinearSVC()),
    ])
    classifier.fit(training_data, training_labels)
    return classifier


def svm_predict(classifier, test_data):
    """
    Predict the labels of the given test data with the given classifier pipeline.
    
    :param sklearn.pipeline.Pipeline classifier: The classifier pipeline with which the labels of the given test data should be predicted
    :param pandas.core.series.Series test_data: The data the given classifier pipeline should predict the labels of
    """
    return classifier.predict(test_data)