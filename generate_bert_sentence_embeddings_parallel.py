from helpers import help_bert_sentence_embeddings, embed
from tqdm import tqdm
from tensorflow.keras.models import Model
import pandas as pd
from pandarallel import pandarallel
import tensorflow_hub as hub
import tensorflow as tf
import bert
import sys
import re

FullTokenizer = bert.bert_tokenization.FullTokenizer

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

max_seq_length = int(data.sentence.str.len().max())

pandarallel.initialize(progress_bar=True)

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

embedding_series = data.sentence.parallel_apply(lambda s: embed(s, tokenizer, max_seq_lenght)[0]).to_numpy()
embeddings = pd.DataFrame.from_records(embedding_series)

if re.search("test", input_data) != None:
    suffix = "_test"
elif re.search("full", input_data) != None:
    suffix = "_full"
else:
    suffix = ""
    
embeddings.to_csv("Data/bert_sentence_embeddings{}.csv".format(suffix), index=False)
