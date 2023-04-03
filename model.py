import tensorflow as tf

def makeModelLSTM(input_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64,input_length=input_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, dropout=0.12)))
    model.add(tf.keras.layers.Dense(27, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'])
    return model

def makeModelGRU(input_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64,input_length=input_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64, dropout=0.12)))
    model.add(tf.keras.layers.Dense(27, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'])
    return model