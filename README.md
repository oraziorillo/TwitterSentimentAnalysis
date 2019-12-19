# TwitterSentimentAnalysis

## Installation setup
in order to work properly, you need to install the following libraries.

Don't know why, sometimes there are warning of retrocompatibility when installing numpy 1.17
and I had to stick to 1.16.4

`conda install nltk gensim spacy`
`pip install pyLDAvis`    Not used actually
`pip install vaderSentiment`   Not used actually
`pip install empath`   not used actually
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

# Naive Bayes 


# Support Vector Machine

# Cleaning primitives
In order to clean the dataset we implemented some functions present in the file
`clean_helpers.py`. 
The cleaning methods allow us to clean the punctuation, numbers, urls, tags, stopwords,
lemmatize words, reduce the number of repeated letters down to 2 (English doesn't have any word with more than 2 repetitions of the same letter).
These primitives can be invoked using the python module `clean_data.py`,
which applies all the primitives chosen from a given list. (in order to show the list,
just type `python3 clean_data.py --help`).

# Word2Vec 
The first approach tried is the Word2Vec approach. It embeds the sentences by averaging 
the word vectors of each word present in every sentence. After having embedded the sentences,
a dense neural network is applied to the sentence vector, which classifies the tweets into
positive and negative class.

The pipeline for this approach is given by the following operations: 
- Clean the tweets 
- Split the dataset into train and test. (Only when testing locally).
- Train the word2vec model (if using a pretrained model, it can be skipped).
- Embed the sentences.
- Train a dense neural network to perform classification into postive and negative class.

In order to run the cleaning primitives, just run the command
`python3 clean_data.py --help` and let you get guided by the command line instructions.

Then, you can decide either to split the training data into train and test, according to whether you want to test it locally
or you want to create the submission (of course, when you create the submission, you want to use the entire dataset
and this part is simply skipped)

In order to divide the dataset into train and test set run

`python3 train_test_split.py --help` 

and follow the instructions. In particular, you can choose the split ratio (we used 0.8), and the output files for the pickles that will
contain the dataframes with sentences and labels (the training and the testing dataframe).
Sentences are shuffled, in order to avoid to have all positive or negative tweets fall into one of the two splits.

Afterwards, you need to create and train the Word2Vec model, in order to be able to perform word embeddings. 
The word embeddings will be used when creating the sentence embeddings: in fact by averaging the word vectors
of each word in one sentence, you may come up with a vector representation of the entire sentence. This is 
possible thanks to the Word2Vec properties of keeping semantically similar words close in the representation space.

In order to train the word2vec model, run

`python3 word2vec_model_constructor.py --help` 

and follow the instructions. You can choose the embedding size, whether to use skip-gram or cbow, the size of the window and the number of epochs.
Or, if you prefer, you can use a pretrained model (we used Google's pretrained model, obtainable here `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit`)
We empirically experienced better results using 5 epochs and word vectors of size 200-300. Using cbow or skip-gram didn't affect results in practice, and nor the window size (which we eventually set to 5, as suggested by the author[1])

When deciding whether to use the pretrained model or not, you have to consider that: Google's model is certainly going to be more accurate,
as it has been trained on a huge dataset (6 billions sentences), but at the same time it is not specialized for the Twitter 
dataset provided for the challenge. Your own model, instead, having been trained especially on the training dataset, is going to be 
probably less accurate than Google's one, but you will have at least all the embeddings for the words in the vocabulary.

We noticed, in fact, that when using Google's model we could only manage to find the representation for about 
60,000 unique words, if compared to more than 350,000 when using a custom model. The fact that Google's model has the representation
for only a limited set of words may turn to be a problem, unless we consider the fact that most of the words that Google's model 
doesn't have are mispelled words, hashtags or words containing special characters, which may not be particulary important. Moreover, about half of the words appear only once in the entire dataset, and aren't therefore particulary informative.

Next step consists in computing the sentence embeddings. Two approaches have been used: the first one is simply taking the 
average of the word vectors that are present in each sentence. This approach, which is possible to apply thanks to the Word2Vec 
properties of mapping semantically similar words into close vectors in the representation space, proved to work: it achieved an accuracy of 82%
on the platform *AI Crowd*.

In order to compute the sentence embedding averaging the word vectors, run 

`python3 create_sentence_representation.py --help` 

and follow the instructions. In particular, you will be able to select the model that you want to use
to compute the word embeddings. Moreover, a `--limit` argument can be provided during testing (it easily explodes with memory consumption,
mainly when working with Google's model which by itself demands several - around 4 - GigaBytes of memory). In this way we were be able to test if the primitives were correct on a smaller fraction of the dataset, before executing it on the full dataset (it may take several minutes).

A different approach for computing the sentence embedding has been used: instead of averaging the word embeddings in every sentence,
we take advantage of the probability measure given by the *chi2* (chi squared). This measure provides a numerical measure to judge whether the presence of a term in a sentence is dependent or independent from a certain label. We may assume, for instance, that the smile ;) will be highly dependent on the presence of the label :). And the same can be said for terms like *love*, *happy* and so on. If a word has a high *chi2* value then, it means that we should give it more *importance* when computing the average, as its presence in the sentence is more likely to be correlated to the presence of a certain label (they are not independent). This has been achieved taking the average of the word embeddings, but weighted by the chi2 value of the words themselves. In contrast to expectations, this model didn't provide good results, assessing at about 77% when performing cross validation to fine tune the hyper-parameters of the classifier. 

You can create sentence embeddings weighted by the *chi2* value using the module 
`python3 create_chi2_sentence_representation.py --help` and following the instructions.

You will be asked to provide a word2vec pretrained model, a dataframe of sentences and a dataframe with the *chi2* value for all the words in our vocabulary.

You can create the chi2 score for all words in the vocabulary of the model chosen by running 
# still need testing, never actually tried it
the module `python3 compute_chi2_for_words.py --help` 

It will output a dataframe, where all words are mapped to their chi2 value.

The sentence representation vectors, then, are fed into a dense neural network which tries to predict the correct labels (1, 0 for positive or negative)
according to the training labels. We used cross validation in order to fine tune the hyperparameters of the neural network,
although given the complexity and the required time to train every model for several epochs, we had to use just one train and test split.

## Need to be converted to python
You can run the classifier by running the notebook `word2vec_nn.ipynb`

# Convolutional Neural Network

A different approach has been attempted in order to perform sentence embedding, which implies the usage of a *Convolutional Neural Network*.

Even in this approach you need to represent words using word vectors. We decided to use as models both a custom one, trained on the dataset provided by us (step `python3 word2vec_model_constructor.py --help`), Google's Word2Vec model [2] and a GloVe pretrained model on a Twitter dataset [3]. All of them proved to work equally good in practice, reaching accuracy from 82% to 83%. 

The approach consists in the following: we embed sentences by representing them as a bidimensional matrix of fixed size, where all the words appearing in the sentence represent one row in the matrix, and the width of the matrix is given by the size of the word embedding chosen (200-300 in our case).

If any sentence is not long enough, which means that some rows would be left empty, then they are padded with either zero vectors or random vectors (again, both approaches didn't prove to work differently in practice, and are therefore to be considered equally good).



# References
- [1]: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
- [2] https://code.google.com/archive/p/word2vec/
- [3] https://nlp.stanford.edu/projects/glove/
