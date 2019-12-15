# TwitterSentimentAnalysis

## Installation setup
in order to work properly, you need to install the following libraries.

Don't know why, sometimes there are warning of retrocompatibility when installing numpy 1.17
and I had to stick to 1.16.4

`conda install nltk gensim spacy`
`pip install pyLDAvis`
`pip install vaderSentiment`
`pip install empath`
`python -m spacy download en`
`python -m nltk.downloader punkt`
`python -m nltk.downloader all-corpora`
`pip install spacy-langdetect`
`conda install -c conda-forge textblob`
`pip install fasttext`
`conda install -c conda-forge tqdm`
`conda install scikit-learn`
`conda install -c anaconda pandas`
`conda install -c anaconda numpy`
`conda install scikit-learn`
`conda install -c conda-forge tqdm`
`conda install gensim`
`sudo apt-get install mysql-server`   these are dependencies of pattern
`sudo apt-get install libmysqlclient-dev`   these are dependencies of pattern
`pip install Pattern`


# Pipeline
The programs works like that:
First, it performs some cleaning primitives for cleaning the sentences present in the dataset.

Then, you can decide either to split the training data into train and test, according to whether you want to test it locally
or you want to create the submission (of course, when you create the submission, you want to use the entire dataset
and this part is simply skipped)

Afterwards, you need to create and train the Word2Vec model, in order to be able to perform word embeddings. 
The word embeddings will be used when creating the sentence embeddings: in fact by averaging the word vectors
of each word in one sentence, you may come up with a vector representation of the entire sentence. This is 
possible thanks to the Word2Vec properties of keeping semantically similar words close in the representation space.

The sentence representation vectors, then, are fed into a neural network which tries to predict the correct labels (1, 0 for positive or negative)
according to the training labels.

