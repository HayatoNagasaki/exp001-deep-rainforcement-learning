from TicTacToe import createDataset
from model import createModel
import numpy as np


if __name__ == '__main__':
    x_train, y_train = createDataset(5000)

    x_train = np.array(x_train).reshape(-1, 9)
    y_train = np.array(y_train).reshape(-1, 9)

    model = createModel(load='model_TicTacToe.h5')
    model.evaluate(x_train, y_train)
    for i in range(10):
        model.fit(x_train, y_train, epochs=50, verbose=0)
        model.evaluate(x_train, y_train)
        model.save('model_TicTacToe.h5')
exit()
