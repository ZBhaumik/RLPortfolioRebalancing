class DQNAgent:
    def __init__(self, state_size, window_size, batch_size):
        self.state_size = state_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.allocation = [1/self.state_size] * self.state_size  # initial allocation
        self.inventory = []  # stocks currently held

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """ Huber loss function
        """
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * clip_delta**2 + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=self.state_size, activation="linear"))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=0.001))
        return model

    def update_target_model(self):
        """ Copies weights from model to target model
        """
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """ Returns action based on current state and epsilon
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.state_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def update_portfolio(self, price, action):
        """ Updates portfolio based on action taken
        """
        # update portfolio allocation
        self.allocation[action] += self.allocation[action] * 0.1
        self.allocation[action] = max(self.allocation[action], 0)  # set lower bound
        self.allocation = [x/sum(self.allocation) for x in self.allocation]  # normalize
        # update inventory
        if action == self.state_size - 1:  # sell all
            self.inventory = []
        else:
            if action in self.inventory:  # sell
                self.inventory.remove(action)
            else:  # buy
                self.inventory.append(action)
    
    def remember(self, state, action, reward, next_state, done):
        """ Stores experiences in memory
        """
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        """ Trains model on experiences stored in memory
        """
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """ Loads weights from file
        """
        self.model.load_weights(name)

    def save(self, name):
        """ Saves weights to file
        """
        self.model.save_weights(name)

    def get_portfolio(self, price):
        """ Returns current portfolio value
        """
        value = 0
        for i in self.inventory:
            value += price[i]
        return value
    def test_model(agent, data, window_size, initial_offset=10000):
        """ Tests model on given data set
        """
        data_length = len(data) - 1
        agent.inventory = []
        state = get_state(data, 0, window_size + 1)
        total_profit = 0
        for t in range(data_length):
            action = agent.act(state)
            agent.update_portfolio(data[t], action)
            next_state = get_state(data, t + 1, window_size + 1)
            reward = (agent.get_portfolio(data[t]) - initial_offset) / initial_offset
            total_profit += reward
            state = next_state
        return total_profit