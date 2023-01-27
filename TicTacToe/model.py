import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
import numpy as np


def createModel(load=''):
    if load != '':
        model = tf.keras.models.load_model(load, compile=False)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        return model

    model = tf.keras.models.Sequential()
    model.add(Dense(32, input_dim=9, activation="sigmoid"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(9, activation="softmax"))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.summary()

    return model


if __name__ == "__main__":
    model = createModel()
