import utils
import tokenization

# required modules
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np

# optional modules
import nltk
import string

# nltk.download('punkt') # only required for the first time
# nltk.download('averaged_perceptron_tagger') # only required for the first time
# nltk.download('stopwords') # only required for the first time

if __name__ == '__main__':

    TEST_SOLUTION_DATA_PATH = "data/test_data_solution.txt"
    TRAIN_DATA_PATH = "data/train_data.txt"
    TOKENIZER_PATH = "tokenizers/tokenizerWP.pickle"

    # read train data set
    df_train = utils.load_data(TRAIN_DATA_PATH)
    print(df_train["genre"])
    print(df_train["genre"][0])

    # read test data set
    df_test = utils.load_data(TEST_SOLUTION_DATA_PATH)
    print(df_test.head())

    #load tokenizer from file
    tokenizer = tokenization.load_tokenizer(TOKENIZER_PATH)
    outFull = tokenizer.encode_batch(df_test['description'])

    print(len(outFull))
    print(outFull[0].tokens)
