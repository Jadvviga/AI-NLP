import utils
import tokenization
import model

# required modules
import tenserflow as tf
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

    # load tokenizer from file
    tokenizer = tokenization.load_tokenizer(TOKENIZER_PATH)
    outFull = tokenizer.encode_batch(df_test['description'])

    target_length = tokenization.encodings_get_length_greater_than(encodings=outFull, percentage=80)
    tokenization.encodings_normalize_length(outFull, target_length=target_length)


    print(df_test['description'][0])
    print(outFull[0].tokens)

    model = model.makeModelLSTM(target_length)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, restore_best_weights=True)

    history = model.fit(data_generator_train, validation_data=data_generator_test, epochs=5)
    model_filename = f"testing_model_{INPUT_SHAPE[0]}_{INPUT_SHAPE[1]}.h5"
    model.save(f"models/{model_filename}")

    utils.make_plots_from_history(history,PLOTS_PATH, model_filename)

    


