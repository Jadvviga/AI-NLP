import utils
import tokenization
import model

# required modules
import tensorflow as tf
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import os

# optional modules
import nltk
import string

#nltk.download('punkt') # only required for the first time
#nltk.download('averaged_perceptron_tagger') # only required for the first time
#nltk.download('stopwords') # only required for the first time

if __name__ == '__main__':
    TEST_SOLUTION_DATA_PATH = "data/test_4categories.txt"

    TOKENIZERS_PATH = "tokenizers"
    PLOTS_PATH = "metrics/plots"
    DATA_PATH = "data"

    train_filename = utils.choose_file_to_load(DATA_PATH)

    # read train data set
    df_train = utils.load_data(os.path.join(DATA_PATH, train_filename))
    # read test data set
    df_test = utils.load_data(TEST_SOLUTION_DATA_PATH)



    # make everything lower case and remove trailing whitespaces
    df_train['description'] = df_train['description'].apply(lambda x: x.lower().strip())
    df_train['genre'] = df_train['genre'].apply(lambda x: x.lower().strip())

    df_test['description'] = df_test['description'].apply(lambda x: x.lower().strip())
    df_test['genre'] = df_test['genre'].apply(lambda x: x.lower().strip())

    # Preprocessing by removing stopwords, punctuation and by doing stemming
    df_train['description'] = df_train['description'].apply(tokenization.remove_stopwords_and_punctuation)
    df_train['description'] = df_train['description'].apply(tokenization.simple_stemmer)

    df_test['description'] = df_test['description'].apply(tokenization.remove_stopwords_and_punctuation)
    df_test['description'] = df_test['description'].apply(tokenization.simple_stemmer)

    tokenizer_filename = utils.choose_file_to_load(TOKENIZERS_PATH)

    labels_dict = utils.get_labels_dict(df_train["genre"])

    labels_train = np.array([labels_dict[genre] for genre in df_train["genre"]])
    labels_test = np.array([labels_dict[genre] for genre in df_test["genre"]])

    Y_test = tf.keras.utils.to_categorical(labels_test, num_classes=len(labels_dict))
    Y_train = tf.keras.utils.to_categorical(labels_train, num_classes=len(labels_dict))

    # load tokenizer from file
    tokenizer = tokenization.load_tokenizer(os.path.join(TOKENIZERS_PATH, tokenizer_filename))
    tokenized_descriptions_train = tokenizer.encode_batch(df_train['description'])
    tokenized_descriptions_test = tokenizer.encode_batch(df_test["description"])

    target_length = tokenization.encodings_get_length_greater_than(encodings=tokenized_descriptions_train,
                                                                   percentage=80)
    tokenization.encodings_normalize_length(tokenized_descriptions_train, target_length=target_length)
    tokenization.encodings_normalize_length(tokenized_descriptions_test, target_length=target_length)

    # print(f"target_length ={target_length}")
    # print(df_test['description'][0])
    # print(tokenized_descriptions_train[0].tokens)
    # print(tokenized_descriptions_train[0].ids)

    model = model.makeModelLSTM_old(target_length, num_categories=len(labels_dict))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5,
                                                restore_best_weights=True)

    X_train = np.array([encoding.ids for encoding in tokenized_descriptions_train])
    X_test = np.array([encoding.ids for encoding in tokenized_descriptions_test])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=7, callbacks=[callback])

    model_filename = f"testing_4categories_oldmodel_{target_length}.h5"
    model.save(f"models/{model_filename}")

    utils.make_plots_from_history(history, PLOTS_PATH, model_filename)
