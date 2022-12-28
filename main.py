#UTILS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import logging
import random

import pandas as pd
import numpy as np

from tqdm import tqdm
import yfinance as yf
import keras.backend as K

# Formats Allocation
def format_allocation(allocation):
    return ' '.join([str(round(a*100, 2))+'%' for a in allocation])

# Formats Currency
def format_currency(price):
    return '${0:.2f}'.format(abs(price))

#UTILS (continued)

def show_train_result(result, val_return, initial_allocation):
    """ Displays training results
    """
    print(result[3])
    if val_return == initial_allocation or val_return == 0.0:
        logging.info('Episode {}/{} - Train Allocation: {}  Val Return: USELESS  Train Loss: {:.4f}'
                    .format(result[0], result[1], format_allocation(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Allocation: {}  Val Return: {}  Train Loss: {:.4f})'
                    .format(result[0], result[1], format_allocation(result[2]), format_currency(val_return), result[3],))

def show_eval_result(model_name, return_, initial_allocation):
    """ Displays eval results
    """
    if return_ == initial_allocation or return_ == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_currency(return_)))

dates = ["2010-01-01","2017-01-01","2017-01-02","2018-01-02","2018-01-03", "2019-01-03"]
#UTILS (continued)

def yfinance_retrieve(stock_names, type):
    type=type*2
    data = {}
    for stock in stock_names:
        df = yf.download(stock, start=dates[type], end=dates[type+1])
        data[stock] = list(df['Adj Close'])
    return data

def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#OPS
import os
import math
import logging

import numpy as np

#OPS
import os
import math
import logging

import numpy as np

def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)

#OPS (continued)

def get_state(data, t, n_days, stock):
    """Returns an n-day state representation ending at time t for a specific stock
    """
    d = t - n_days + 1
    block = data[stock][d: t + 1] if d >= 0 else -d * [data[stock][0]] + data[stock][0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    res.append(sigmoid(block[-1] - block[-2]))  # add last difference
    return np.array([res])

#METHODS

import os
import logging
import numpy as np
from collections import deque
import dill as pickle

#METHODS (continued)

#METHODS

import os
import logging
import numpy as np
from collections import deque
import dill as pickle

#METHODS (continued)

def train_model(agent, episode, data, stock, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data[stock]) - 1
    batch_size = batch_size

    for e in range(ep_count + 1):
        print("Episode " + str(e) + "/" + str(ep_count))
        state = get_state(data, 0, window_size, stock)

        total_profit = 0
        agent.inventory = []

        for t in range(data_length):
            action = agent.act(state)

            # sit
            next_state = get_state(data, t + 1, window_size, stock)
            reward = 0

            if action == 1:  # buy
                agent.inventory.append(data[stock][t])
                print("Buy: " + format_currency(data[stock][t]))

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[stock][t] - bought_price, 0)
                total_profit += data[stock][t] - bought_price
                print("Sell: " + format_currency(data[stock][t]) + " | Profit: " + format_currency(data[stock][t] - bought_price))

            done = True if t == data_length - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + format_currency(total_profit))
                print("--------------------------------")

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 10 == 0:
            agent.model.save("models/model_ep" + str(e))

    return total_profit

def evaluate_model(agent, data, stock, window_size, epoch_count, initial_investment, train, model_name):
    net_returns = []

    for e in range(epoch_count):
        total_profit = 0
        state = get_state(data, 0, window_size + 1, stock)

        for t in range(len(data[stock]) - 1):
            action = agent.act(state)

            # apply action
            done = False
            if action == 1:
                agent.inventory.append(data[stock][t])
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[stock][t] - bought_price, 0)
                total_profit += data[stock][t] - bought_price
                done = True
            else:
                reward = 0

            if t == len(data[stock]) - 2:
                done = True

            next_state = get_state(data, t + 1, window_size + 1, stock)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                net_returns.append(total_profit)
                break
        if train:
            agent.expReplay(32)

    return net_returns

def run_experiment(stock_names, window_size, epoch_count, initial_investment, model_name="", train=False):
    data = yfinance_retrieve(stock_names, 0)
    val_data = yfinance_retrieve(stock_names, 1)

    if not os.path.isdir("models"):
        os.makedirs("models")

    switch_k_backend_device()

    try:
        agent = pickle.load(open("models/" + model_name + ".pkl", "rb"))
    except Exception as err:
        agent = DQNAgent(window_size)

    if train:
        train_model(agent, 1, data, stock_names[0], ep_count=epoch_count, batch_size=32, window_size=window_size)
        agent.model.save("models/" + model_name + ".h5")
        with open("models/" + model_name + ".pkl", "wb") as f:
            pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    net_returns = evaluate_model(agent, data, stock_names[0], window_size, epoch_count, initial_investment, False, model_name)
    val_returns = evaluate_model(agent, val_data, stock_names[0], window_size, epoch_count, initial_investment, False, model_name)

    # plot the returns
    import matplotlib.pyplot as plt
    plt.plot(net_returns, label="Train")
    plt.plot(val_returns, label="Validation")
    plt.legend()
    plt.show()

    return sum(net_returns) / len(net_returns), sum(val_returns) / len(val_returns)

import random
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size=3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, alpha=0.001, alpha_decay=0.01, gamma=0.95):
        self.state_size = state_size
        print(state_size)
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])



if __name__ == "__main__":
    stock_names = ['AAPL', 'TSLA', 'GOOGL', 'AMZN']
    window_size = 10
    epoch_count = 100
    initial_investment = 20000
    model_name = "model_1"

    run_experiment(stock_names, window_size, epoch_count, initial_investment, model_name, train=True)