from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def naive_bayes_fit(training_data, training_labels):
    #construct a multinomial naive Bayes classifier based fitted on the given training data and the corresponding class labels
    classifier = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 4))), #add arguments to CountVectorizer for more efficiency
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    classifier.fit(training_data, training_labels)
    return classifier


def naive_bayes_predict(classifier, test_data):
    #predict the labels of the given test data with the given classifier
    return classifier.predict(test_data)