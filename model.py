import tensorflow as tf


def makeModelLSTM_old(input_length, num_categories):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=input_length, mask_zero=True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, dropout=0.5)))
    model.add(tf.keras.layers.Dense(num_categories, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'], optimizer='adam')
    return model

def makeModelLSTM2layers(input_length, num_categories):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=input_length, mask_zero=True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16, dropout=0.4, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16, dropout=0.4)))
    model.add(tf.keras.layers.Dense(num_categories, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'], optimizer='adam')
    return model


#todo add to emebdding layer mask_zero=True

def makeModelGRU(input_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=input_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64, dropout=0.12)))
    model.add(tf.keras.layers.Dense(27, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'])
    return model
