from TicTacToe import createDataset
from model import createModel
import numpy as np


def train(use_model='', n_dataset=5000, epochs=100, subepochs=50):
    x_train, y_train = createDataset(n_dataset)

    x_train = np.array(x_train).reshape(-1, 9)
    y_train = np.array(y_train).reshape(-1, 9)

    model = createModel(load=use_model)
    model.evaluate(x_train, y_train)
    for i in range(epochs):
        model.fit(x_train, y_train, epochs=subepochs, verbose=0)
        model.evaluate(x_train, y_train)
        model.save(use_model)


if __name__ == '__main__':
    train(use_model='model_TicTacToe.h5', n_dataset=5000, epochs=10, subepochs=50)
exit()
