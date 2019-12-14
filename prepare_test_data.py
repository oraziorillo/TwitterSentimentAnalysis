import pandas as pd
import argparse


from clean_helpers import *



parser = argparse.ArgumentParser(description='Builds sentence representation using word vectors.')

parser.add_argument('--test_data_input',
                    required=True,
                    help='test file with sentences to classify')

parser.add_argument('--output_df',
                    required=True,
                    help='test file with sentences to classify')

parser.add_argument('--clean_methods', 
                    help='cleaning methods, can choose among: \n' + 
                        "clean_new_line\n" +
                        "lowercase\n" + "lemmatize, remove_stopwords" +
                        "clean_punctuation clean_tags remove_numbers" +
                        "remove_saxon_genitive, gensim_simple, more_than_double_rep",
                    nargs='+')

args = parser.parse_args()

# Cleaning methods defined in clean_helpers.py
clean = {
    "clean_new_line": clean_new_line,
    "lowercase": lowercase,
    "lemmatize": lemmatize,
    "remove_stopwords": remove_stopwords,
    "translate": perform_translation,
    "clean_punctuation": clean_punctuation,
    "clean_tags" : clean_tags,
    "remove_numbers": remove_numbers,
    "remove_saxon_genitive": remove_saxon_genitive,
    "gensim_simple": gensim_clean,   # not a good idea to use it I think! It cleans everything which is not alphabetic (special char, numbers and so on)
    "more_than_double_rep": clean_more_than_double_repeated_chars,
    "clean_spelling": clean_spelling
}


cleaning_options = args.clean_methods

df_test = []
with open(args.test_data_input, 'r') as f:
    for l in f:
        id_ = l.split(",")[0]
        # it is a csv, but you have to keep other commas (only the first one is relevant)
        sentence = ",".join(l.split(",")[1:])
        df_test.append({
            "label": int(id_),
            "sentence": sentence
        })
df_test = pd.DataFrame(df_test)
df_test.head()


for clean_option in cleaning_options:
    df_test = clean[clean_option](df_test)
    print(clean_option)
    print(df_test.head())
    print("################################\n\n")


print("Save df to pickle")
df_test.to_pickle(args.output_df)

