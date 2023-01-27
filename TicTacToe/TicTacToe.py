import numpy as np
import random


class TicTacToe:
    def __init__(self):
        # 0 = '', 1 = ✕, 2 = ◯
        self.states = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.__states = ['', '', '', '', '', '', '', '', '']
        self.action_space_n = len(self.states)

    def step(self, action, player):
        if self.states[action] == 0:
            # action = 0-8, player = 1 or 2
            self.states[action] = player
            error = False
        else:
            error = True

        return self.states, self.evaluate(), error  # states, reward, error

    def evaluate(self):
        # 0 = not finished, 1 = ✕ wins, 2 = ◯ wins, 3 = draw
        # check rows
        for i in [0, 3, 6]:
            if self.states[i] == self.states[i+1] == self.states[i+2] != 0:
                return self.states[i]
        # check columns
        for i in range(3):
            if self.states[i] == self.states[i+3] == self.states[i+6] != 0:
                return self.states[i]
        # check diagonals
        if self.states[0] == self.states[4] == self.states[8] != 0:
            return self.states[0]
        if self.states[2] == self.states[4] == self.states[6] != 0:
            return self.states[2]
        # check draw
        if 0 not in self.states:
            return 3
        # game is not finished
        return 0

    def render(self):
        for i in range(len(self.states)):
            if self.states[i] == 0:
                self.__states[i] = ' '
            elif self.states[i] == 1:
                self.__states[i] = '✕'
            elif self.states[i] == 2:
                self.__states[i] = '◯'
        print(f'-------------')
        print(f'| {self.__states[0]} | {self.__states[1]} | {self.__states[2]} |')
        print(f'-------------')
        print(f'| {self.__states[3]} | {self.__states[4]} | {self.__states[5]} |')
        print(f'-------------')
        print(f'| {self.__states[6]} | {self.__states[7]} | {self.__states[8]} |')
        print(f'-------------')


def run(render=False, use_model=''):
    if use_model != '':
        from model import createModel
        model = createModel(load=use_model)
    env = TicTacToe()
    human = 1  # ✕
    cpu = 2  # ◯
    reward = 0
    X = []  # list of ovservations
    A = []  # list of actions

    while reward == 0:
        error = True
        # human
        while error and reward == 0:
            a1 = random.randint(0, env.action_space_n - 1)
            states, reward, error = env.step(a1, human)
            if not error:
                X.append([_ for _ in states])
                A.append(-1)

        error = True
        # cpu
        while error and reward == 0:
            if use_model != '':
                a2 = np.argmax(model.predict(np.array([states]), verbose=0))
            else:
                a2 = random.randint(0, env.action_space_n - 1)
            states, reward, error = env.step(a2, cpu)
            if not error:
                X.append([_ for _ in states])
                A.append(a2)

    if render:
        env.render()

        if reward == 1:
            print('✕ wins (human)')
        elif reward == 2:
            print('◯ wins (cpu)')
        elif reward == 3:
            print('draw')

    return env, X, A, reward


def createDataset(n=1000, use_model=''):
    x_train = []
    y_train = []

    for i in range(n):
        env, X, A, reward = run(render=False, use_model=use_model)

        if reward == 2:
            for i, x in enumerate(X[0:-1]):
                if A[i] == -1 and x not in x_train:
                    x_train.append(x)

                    vec = [0] * env.action_space_n
                    vec[A[i + 1]] = 1
                    y_train.append(vec)

    return x_train, y_train


if __name__ == "!__main__":
    x_train, y_train = createDataset(1000)

    x_train = np.array(x_train).reshape(-1, 9)
    y_train = np.array(y_train).reshape(-1, 9)

    print(len(x_train))
    print(len(y_train))

if __name__ == "__main__":
    human_wins = 0
    cpu_wins = 0

    for i in range(100):
        env, X, A, reward = run(render=False, use_model='TicTacToe/model_TicTacToe.h5')
        # env, X, A, reward = run(render=False)

        if reward == 1:
            human_wins += 1
        elif reward == 2:
            cpu_wins += 1

    print(f'human wins: {human_wins}, cpu wins: {cpu_wins}')
    exit()
