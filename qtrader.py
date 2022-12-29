import numpy as np
import pandas as pd
from collections import deque
import random
import yfinance as yf

dates = [
    "2010-01-01",
    "2019-06-06",
]

def yfinance_retrieve(stock_names, start_date, end_date):
    stock_data = pd.DataFrame()
    for stock in stock_names:
        df = yf.download(stock, start=start_date, end=end_date)
        df = df[["Adj Close"]]
        df.columns = [stock]
        stock_data = stock_data.join(df, how="outer")
    return stock_data

import numpy as np
import pandas as pd
from collections import deque
import random


import numpy as np
import pandas as pd
import tensorflow as tf

class QTrader:
  def __init__(self, stock_data, initial_portfolio, funds, epsilon=0.5, alpha=0.5, gamma=0.9, decay=0.99):
    self.stock_data = stock_data
    self.initial_portfolio = initial_portfolio
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.decay = decay
    self.portfolio = initial_portfolio.copy()
    self.memory = deque(maxlen=100000)
    self.buy_sell_history = []
    self.funds = funds

    self.state_dim = len(self.portfolio.keys())
    self.action_dim = sum(self.portfolio.values())  # Set action_dim to the total number of shares that can be bought or sold
    self.hidden_dim = 32

    self.model = self.build_model()
    self.target_model = self.build_model()  # Add target model


  def build_model(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(self.hidden_dim, input_shape=(self.state_dim,), activation="relu"),
        tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
        tf.keras.layers.Dense(self.action_dim, activation="linear")
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.alpha))
    return model

  def get_state(self, t):
    self.stock_prices = {stock: self.stock_data[stock][t] for stock in self.portfolio.keys()}
    return np.array(list(self.stock_prices.values()))

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def choose_actions(self, state, t):
    self.epsilon *= self.decay
    actions = []
    funds=self.funds
    keys = list(self.portfolio.keys())  # Convert keys to a list
    total_cost = 0  # Initialize the total cost to 0
    for i, stock in enumerate(keys):  # Iterate over the list of keys
      if np.random.uniform() < self.epsilon:
        low = min(self.portfolio.values())
        high = max(self.portfolio.values())
        action = np.random.randint(-low, high)
      else:
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values[0])
      # Calculate the cost of the selected action
      cost = self.stock_data[stock][t] * action
      total_cost += cost  # Add the cost to the total cost
      # If the total cost exceeds the available funds, set action to the maximum number of shares that can be bought
      if total_cost > funds:
        action = (funds - (total_cost - cost)) // self.stock_data[stock][t]
      low = -self.portfolio[stock]  # Use stock to access the corresponding value in self.portfolio
      high = self.portfolio[stock]
      action = np.clip(action, low, high)
      # Ensure that action is within the valid range for the target_vec array
      action = np.clip(action, -self.action_dim, self.action_dim-1)
      actions.append(action)
    actions = [int(x) for x in actions]
    return actions

  def take_actions(self, actions, t):
    reward = 0
    for i, stock in enumerate(self.portfolio.keys()):
      self.portfolio[stock] += actions[i]
      self.funds -= self.stock_data[stock][t] * actions[i]  # Update the funds variable
    reward = sum([self.stock_data[stock][t] * actions[i] for i, stock in enumerate(self.portfolio.keys())])
    return reward

  def update_memory(self, t):
    state = self.get_state(t)
    actions = self.choose_actions(state, t)
    reward = self.take_actions(actions, t)
    next_state = self.get_state(t+1)
    self.memory.append((state, actions, reward, next_state))
    self.buy_sell_history.append((actions, reward))

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    states = np.array([t[0] for t in minibatch])
    actions = np.array([t[1] for t in minibatch]).reshape(-1,1)
    rewards = np.array([t[2] for t in minibatch])
    next_states = np.array([t[3] for t in minibatch])

    # Compute targets using target model
    targets = rewards + self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1)
    target_vec = self.model.predict(states, verbose=0)
    target_vec[np.arange(batch_size), actions] = targets
    self.model.fit(states, target_vec, epochs=1, verbose=0)

  def trade(self, num_episodes, batch_size):
    for episode in range(num_episodes):
      self.stock_data = train_data
      self.portfolio = self.initial_portfolio.copy()
      self.buy_sell_history = []
      self.funds=0
      for t in range(len(self.stock_data["AAPL"]) - 1):
        print(t)
        self.update_memory(t)
        if (len(self.memory) > batch_size) and (t%batch_size==0):
          self.replay(batch_size)
          self.update_target_model()  # Update target model
      self.total_profit = sum(self.portfolio.values()) - sum(self.initial_portfolio.values())
      print(f"Episode {episode+1}: Train Profit = {self.total_profit}")
      #VALIDATION LOOP
      self.stock_data = val_data
      self.portfolio = self.initial_portfolio.copy()
      self.funds=0
      state = self.get_state(0)
      for t in range(len(self.stock_data.index)-1):
        actions = self.choose_actions(state, t)
        reward = self.take_actions(actions, t)
        state = self.get_state(t+1)
      self.total_profit = sum(self.portfolio.values()) - sum(self.initial_portfolio.values())
      print(f"Episode {episode+1}: Val Profit = {self.total_profit}")




  def test(self):
    self.epsilon = 0.0  # Set epsilon to 0 to disable exploration during test
    state = self.get_state(0)
    self.funds=0
    self.portfolio = self.initial_portfolio.copy()  # Reset portfolio to initial values
    for t in range(len(self.stock_data.index)-1):
      actions = self.choose_actions(state, t)
      reward, done = self.take_actions(actions, t)
      state = self.get_state(t+1)
      if done:
        break
    self.total_profit = sum(self.portfolio.values()) - sum(self.initial_portfolio.values())
    print(f"Test Profit = {self.total_profit}")

import keras.backend as K
import os
def switch_k_backend_device():
    if K.backend() == "tensorflow":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

stock_data = yfinance_retrieve(["AAPL", "GOOGL", "MSFT"], dates[0], dates[1])

# Split data into training and testing sets
split_index = int(0.8 * len(stock_data))
train_data = stock_data.iloc[:split_index]
test_data = stock_data.iloc[split_index:]
split_index = int(0.5 * len(test_data))
val_data = test_data.iloc[:split_index]
test_data = test_data.iloc[split_index:]

# Set initial portfolio
initial_portfolio = {"AAPL": 10, "GOOGL": 5, "MSFT": 15}

switch_k_backend_device()
# Initialize StockTrader object
trader = QTrader(train_data, initial_portfolio, funds=0)

# Train the trader
trader.trade(num_episodes=10, batch_size=32)

# Test the trader on unseen data
trader.stock_data = test_data  # Set trader's stock data to test data
trader.test()