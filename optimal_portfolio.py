import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import random
import yfinance as yf

dates = [
    "2010-01-01",
    "2019-06-06",
]

batch_size=32

def yfinance_retrieve(stock_names, start_date, end_date):
    stock_data = pd.DataFrame()
    for stock in stock_names:
        df = yf.download(stock, start=start_date, end=end_date)
        df = df[["Adj Close"]]
        df.columns = [stock]
        stock_data = stock_data.join(df, how="outer")
    return stock_data


class QTrader:
    def __init__(
        self,
        stock_data,
        initial_portfolio,
        funds,
        epsilon=0.5,
        alpha=0.5,
        gamma=0.9,
        decay=0.99,
    ):
        self.stock_data = stock_data
        self.initial_portfolio = initial_portfolio
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay = decay
        self.portfolio = initial_portfolio.copy()
        self.initial_portfolio_value = funds
        self.memory = deque(maxlen=100000)
        self.buy_sell_history = []
        self.funds = funds

        self.state_dim = (
            len(self.portfolio.keys()) * 3
        )  # Add 2 for market trend and volatility
        self.action_dim = len(
            self.portfolio.keys()
        )  # Set action_dim to the number of stocks in the portfolio
        self.hidden_dim = 32

        self.model = self.build_model()
        self.target_model = self.build_model()  # Add target model

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.hidden_dim, input_shape=(self.state_dim,), activation="relu"
                ),
                tf.keras.layers.Dense(self.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.action_dim, activation="linear"),
            ]
        )
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.alpha))
        return model

    def get_state(self, t):
            # Calculate market trend and volatility for each stock
            stock_trends = {}
            stock_volatilities = {}
            for stock in self.portfolio.keys():
                stock_trends[stock]=0
                stock_volatilities[stock]=0
                if(t!=0):
                    # Calculate market trend
                    stock_trends[stock] = (self.stock_data[stock][t] - self.stock_data[stock][t-1]) / self.stock_data[stock][t-1]
                    # Calculate volatility
                    stock_volatilities[stock] = np.std([self.stock_data[stock][i] for i in range(max(t-10,0), t)])

            self.stock_prices = {stock: self.stock_data[stock][t] for stock in self.portfolio.keys()}
            return np.array(
                list(self.stock_prices.values())
                + list(stock_trends.values())
                + list(stock_volatilities.values())
            )

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_actions(self, state, t):
        self.epsilon *= self.decay
        actions = []
        keys = list(self.portfolio.keys())  # Convert keys to a list
        if np.random.uniform() < self.epsilon:
            actions = [
                np.random.randint(-min(self.portfolio[stock], self.funds // self.stock_data[stock][t]), min(self.funds // self.stock_data[stock][t] + 1, self.portfolio[stock] + 1))
                for stock in keys
            ]
        else:
            # Choose the action with the highest predicted value for each stock
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            actions = [np.argmax(q_values[0][i:i+1]) - self.portfolio[keys[i]]for i in range(self.action_dim)]
            stock_returns = {}
            for i, stock in enumerate(keys):
                q_values = q_values.flatten()
                stock_returns[stock] = q_values[int(min(actions[i] + self.portfolio[stock] + i * self.action_dim, len(q_values) - 1))]
            # Sort the stocks by expected return in descending order
            sorted_stocks = sorted(stock_returns.items(), key=lambda x: x[1], reverse=True)

            # Allocate funds to the top performing stocks
            top_stocks = sorted_stocks[: int(len(sorted_stocks) / 2)]
            for i, (stock, _) in enumerate(top_stocks):
                # Sell stocks with lower expected returns
                if i > 0:
                    if(self.portfolio[stock]>0):
                        self.funds += self.stock_data[stock][t] * self.portfolio[stock]
                        self.portfolio[stock] = 0
                # Buy more of the top performing stocks
                if(self.funds>=(self.stock_data[stock][t]*self.portfolio[stock])):
                    self.portfolio[stock] += self.funds // self.stock_data[stock][t]
                    self.funds -= self.stock_data[stock][t] * self.portfolio[stock]
        return actions

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def get_portfolio_value(self, t):
        portfolio_value = 0
        for stock in self.portfolio.keys():
            portfolio_value += self.stock_data[stock][t] * self.portfolio[stock]
        return portfolio_value

    def act(self, state, t):
        actions = self.choose_actions(state, t)
        self.buy_sell_history.append(actions)
        portfolio_value = self.get_portfolio_value(t)
        dValue = portfolio_value - self.initial_portfolio_value
        return dValue

    def replay(self, batch_size):
        # Sample a batch of experiences from the memory
        minibatch = random.sample(self.memory, batch_size)

        # Extract the states, actions, rewards, and next states from the minibatch
        states = [example[0] for example in minibatch]
        actions = [example[1] for example in minibatch]
        rewards = [example[2] for example in minibatch]
        next_states = [example[3] for example in minibatch]
        dones = [example[4] for example in minibatch]

        # Predict the value of the next states using the target model
        next_q = self.target_model.predict(np.array(next_states), verbose=0)

        # Calculate the target values for the Q-learning update
        targets = []
        for i in range(batch_size):
            if dones[i]:
                targets.append(rewards[i])
            else:
                targets.append(rewards[i] + self.gamma * np.amax(next_q[i]))

        # Update the model's weights using the target values and the Q-learning formula
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)


if __name__ == "__main__":
    stock_names = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    stock_data = yfinance_retrieve(stock_names, dates[0], dates[1])

    # Split the stock data into training, validation, and test sets
    train_data = stock_data[: int(0.7 * stock_data.shape[0])]
    val_data = stock_data[
        int(0.7 * stock_data.shape[0]) : int(0.85 * stock_data.shape[0])
    ]
    test_data = stock_data[int(0.85 * stock_data.shape[0]) :]

    initial_portfolio = {stock: 0 for stock in stock_names}
    funds = 10000

    q_trader = QTrader(stock_data, initial_portfolio, funds)
    q_trader.update_target_model()

    # Train the model on multiple episodes of the training set
    num_episodes = 10  # Set the number of episodes to train on
    for episode in range(num_episodes):
        print(f"Episode {episode+1}")
        for t in range(train_data.shape[0]):
            print(t)
            state = q_trader.get_state(t)
            portfolio_value_change = q_trader.act(state, t)
            reward = portfolio_value_change / q_trader.initial_portfolio_value
            next_state = q_trader.get_state(t + 1)
            done = False
            if t == train_data.shape[0] - 2:
                done = True
            q_trader.remember(
                state, q_trader.buy_sell_history[-1], reward, next_state, done
            )
            if (len(q_trader.memory) > batch_size) and (t%batch_size==0):
                q_trader.replay(batch_size)

        # Calculate the return for the training set
        train_return = q_trader.get_portfolio_value(train_data.shape[0]-1)/q_trader.initial_portfolio_value
        print(q_trader.funds)
        print(q_trader.portfolio)
        print(f"Training return: {train_return}")

        # Calculate the return for the validation set
        #THIS IS WRONG. SO IS THE TEST METHOD.
        val_return = 0
        for t in range(val_data.shape[0]):
            state = q_trader.get_state(t)
            portfolio_value_change = q_trader.act(state, t)
            reward = portfolio_value_change / q_trader.initial_portfolio_value
            val_return += reward
        print(f"Validation return: {val_return}")

    # Test the model on the test set
    test_return = 0
    for t in range(test_data.shape[0]):
        state = q_trader.get_state(t)
        portfolio_value_change = q_trader.act(state, t)
        reward = portfolio_value_change / q_trader.initial_portfolio_value
        test_return += reward
    print(f"Test return: {test_return}")