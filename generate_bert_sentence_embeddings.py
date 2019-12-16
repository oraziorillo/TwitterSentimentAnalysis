from helpers import help_bert_sentence_embeddings, get_ids, get_masks, get_segments
from tqdm import tqdm
from tensorflow.keras.models import Model
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import bert
import sys
import re


if len(sys.argv) != 2:
    print("\n\tWrong number of arguments")
    help_bert_sentence_embeddings()
    sys.exit(0)
    
try:
    input_data = sys.argv[1]
    data = pd.read_csv(input_data)
except:
    print("\n\tNot valid input filepath")
    help_bert_sentence_embeddings()
    sys.exit(0)


def embed(s):
    # Tokenize the sentence 
    stokens = tokenizer.tokenize(s)
    # Add separators according to the paper
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    # Generate the inputs of the model by using tokens
    input_ids = get_ids(stokens, tokenizer, max_seq_length)
    input_masks = get_masks(stokens, max_seq_length)
    input_segments = get_segments(stokens, max_seq_length)
    # Generate Embeddings using the pretrained model
    pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])
    return pool_embs


max_seq_length = int(data.sentence.str.len().max())

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    
FullTokenizer = bert.bert_tokenization.FullTokenizer
tokenizer = FullTokenizer(vocab_file, do_lower_case)

tqdm.pandas()

embedding_series = data.sentence.progress_apply(lambda s: embed(s)[0]).to_numpy()
embeddings = pd.DataFrame.from_records(embedding_series)

embeddings.to_csv("Data/bert_sentence_embeddings_full.csv".format(pn), index=False)



