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
`python -m textblob.download_corpora`


# Word2Vec 
The programs works like that:
First, it performs some cleaning primitives for cleaning the sentences present in the dataset.

In order to run the cleaning primitives, just run the command
`python3 clean_data.py --help` and let you get guided by the command line instructions.

Then, you can decide either to split the training data into train and test, according to whether you want to test it locally
or you want to create the submission (of course, when you create the submission, you want to use the entire dataset
and this part is simply skipped)

In order to divide the dataset into train and test set run

`python3 train_test_split.py --help` 

and follow the instructions. In particular, you can choose the split ratio (we used 0.8), and the output files for the pickles that
contain the dataframes with sentences and labels (the training and the testing dataframe).

Afterwards, you need to create and train the Word2Vec model, in order to be able to perform word embeddings. 
The word embeddings will be used when creating the sentence embeddings: in fact by averaging the word vectors
of each word in one sentence, you may come up with a vector representation of the entire sentence. This is 
possible thanks to the Word2Vec properties of keeping semantically similar words close in the representation space.

In order to train the word2vec model, run

`python3 word2vec_model_constructor.py --help` 

and follow the instructions. You can choose the embedding size, whether to use skip-gram or cbow, the size of the window and the number of epochs.
Or, if you prefer, you can use a pretrained model (we used Google's pretrained, obtainable here `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit`)

When deciding whether to use the pretrained model or not, you have to consider that: Google's model is certainly going to be more accurate,
as it has been trained on a huge dataset (6 billions sentences), but at the same time it is not specialized for the Twitter 
dataset provided for the challenge. Your own model, instead, having been trained especially on your dataset, is going to be 
probably less accurate than Google's one, but you will have at least all the words in your vocabulary.

We noticed, in fact, that when using Google's model, we could only manage to find the representation for about 
60,000 unique words, if compared to more than 350,000 when using a custom model. The fact that Google's model has the representation
for only a limited set of words may turn to be a problem, unless we consider the fact that most of the words that Google's model 
doesn't have are mispelled words, hashtags or words containing special characters, which may not be particulary important, if you consider
that about half of the unique words in the training set are repeated just once (and therefore are not particulary informative).

Next step consists in computing the sentence embedding. Two approaches have been used: the first one is simply taking the 
average of the word vectors that are present in each sentence. This approach, which is possible to apply thanks to the Word2Vec 
properties of mapping semantically similar words into close vectors in the representation space, proved to work: it achieved an accuracy of 82%
on the platform *AI Crowd*.

In order to compute the sentence embedding averaging the word vectors, run 

`python3 create_sentence_representation.py --help` 

and follow the instructions. In particular, you will be able to select the model that you want to use
to compute the word embeddings. Moreover, a `--limit` argument can be provided during testing (it easily explodes with memory consumption,
mainly when working with Google's model which by itself demands several - around 3 - GigaBytes of memory).

A different approach for computing the sentence embedding has been used: instead of averaging the word embeddings in every sentence,
we take advantage of the probability measure given by the *chi2* (chi squared). This measure, is computed over the entire dataset using
the formula ***, and provides a numerical measure to judge whether the presence of a term in a sentence is dependent or independent from 
a certain label. We may assume, for instance, that the smile ;) will be highly dependent on the presence of the label :). And the same 
can be said for terms like *love*, *happy* and so on. If a word then has a high chi2 value, it means 


The sentence representation vectors, then, are fed into a dense neural network which tries to predict the correct labels (1, 0 for positive or negative)
according to the training labels. We used cross validation in order to fine tune the hyperparameters of the neural network,
although given the complexity and the required time to train every model for several epochs, we had to use just one train and test split.



