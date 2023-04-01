import utils

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


TEST_SOLUTION_DATA_PATH = "data/test_data_solution.txt"
TRAIN_DATA_PATH = "data/train_data.txt"

# read train data set
df_train = utils.load_data(TRAIN_DATA_PATH)
#print(df_train["genre"])
#print(df_train["genre"][0])

# read test data set
df_test = utils.load_data(TEST_SOLUTION_DATA_PATH)
#print(df_test.head())

