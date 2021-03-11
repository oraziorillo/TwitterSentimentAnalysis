#### Introduction

This is a Sentiment Analyzer based on a LSTM neural network able to predict whether the sentiment of a sentence is 'joyful' or 'sad'.

#### Setup

Before starting with the training, open the folder _lstm_  and follow these steps to setup your environment

- ensure that you have a folder named _Data_ in which there are three datasets: 

  a) the dataset of the positive sentences as _train_pos_full.txt_ 

  b) the dataset of the negative sentences as _train_neg_full.txt_

  c) the dataset of the sentences for which the prediction should be performed as _test_data.txt_

  NOTE: these should be simple txt files in which each line is a data sample containing the sentence and the sentiment separated by a comma (1 for 'positive', -1 for 'negative), while the test set trivially should be unlabeled

- ensure to have an empty folder named trained_models

- open a terminal and run the command:

  ```{r, engine='bash', count_lines}
  $ pip -r install requirements.txt
  ```

#### Training

Once you are done with this, we are ready to go. 

1. open a shell

2. run the command:

   ```{r, engine='bash', count_lines}
   $ python3 prepare_datasets.py
   ```

   This will clean your datasets and will prepare them to be automatically analyzed

3. run the command:

   ```{r, engine=&#39;bash&#39;, count_lines}
   $ python3 train_model.py -i [ITERATIONS] -ep [EPOCHS]
   ```

   and substitute _[ITERATIONS]_ with the number of trainings you want to perform and _[EPOCHS]_ with the number of epochs per iteration. There are also other options to personalize the training, that will be shown to you if you use the _-h_ option, but the default ones are the ones with which we got the best results so far. 

#### Predictions generation

Finally generate predictions for the preprocessed test set with the command:

```{r, engine=&#39;bash&#39;, count_lines}
$ python3 generate_predictions.py -m [MODEL_NAME]
```

and substitute _[MODEL_NAME]_ with the name of the model you want to use to generate the prediction on the test set. You have to specify only the name of the model, without the extension.
