import pandas as pd

# -------------------------------------------------------------------------------------------------------------------
# Bert stuff
# -------------------------------------------------------------------------------------------------------------------

def help_bert_sentence_embeddings():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 bert_sentence_embeddings.py <sentences_to_embed.csv>")
    

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


# -------------------------------------------------------------------------------------------------------------------
# LSTM stuff
# -------------------------------------------------------------------------------------------------------------------

def help_train_lstm():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 help_train_lstm.py <number_of_epochs>")
    
def help_keep_training_lstm():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 help_keep_training_lstm.py <model_name> <number_of_epochs>")
    
def help_generate_predictions():
    print("\n\tTo use this tool use the command:\
            \n\t\tpython3 generate_predictions.py <model_name>")
