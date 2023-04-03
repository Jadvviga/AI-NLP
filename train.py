import utils
import tokenization
import model

# required modules
import tensorflow as tf
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
    PLOTS_PATH = "metrics/plots"

    # read train data set
    df_train = utils.load_data(TRAIN_DATA_PATH)
    # print(df_train["genre"])
    # print(df_train["genre"][0])

    # read test data set
    df_test = utils.load_data(TEST_SOLUTION_DATA_PATH)
    # print(df_test.head())

    labels_dict = utils.get_labels_dict(df_train["genre"])

    labels_train = np.array([labels_dict[genre] for genre in df_train["genre"]])
    labels_test = np.array([labels_dict[genre] for genre in df_test["genre"]])

    Y_test = tf.keras.utils.to_categorical(labels_test, num_classes=len(labels_dict))
    Y_train = tf.keras.utils.to_categorical(labels_train, num_classes=len(labels_dict))

    # load tokenizer from file
    tokenizer = tokenization.load_tokenizer(TOKENIZER_PATH)
    tokenized_descriptions_train = tokenizer.encode_batch(df_train['description'])
    tokenized_descriptions_test = tokenizer.encode_batch(df_test["description"])

    target_length = tokenization.encodings_get_length_greater_than(encodings=tokenized_descriptions_train,
                                                                   percentage=80)
    tokenization.encodings_normalize_length(tokenized_descriptions_train, target_length=target_length)
    tokenization.encodings_normalize_length(tokenized_descriptions_test, target_length=target_length)

    print(f"target_length ={target_length}")
    print(df_test['description'][0])
    print(tokenized_descriptions_train[0].tokens)
    print(tokenized_descriptions_train[0].ids)

    model = model.makeModelLSTM(target_length)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5,
                                                restore_best_weights=True)

    X_train = np.array([encoding.ids for encoding in tokenized_descriptions_train])
    X_test = np.array([encoding.ids for encoding in tokenized_descriptions_test])

    print(X_train.shape)
    print(X_train)
    print(Y_train.shape)
    print(Y_train)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=2, callbacks=[callback])

    model_filename = f"testing_model2_{target_length}.h5"
    model.save(f"models/{model_filename}")

    utils.make_plots_from_history(history, PLOTS_PATH, model_filename)
