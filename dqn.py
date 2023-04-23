import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

# Build a neural network for your DQN agent
def build_model(action_space,obs_space):
    # Build and compile Q-network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(obs_space.shape)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Define the DQN agent
class DQNAgent:
    def __init__(self,action_space,obs_space):
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        self.action_space = action_space
        self.memory = deque(maxlen=10000)
        self.model = build_model(action_space.n,obs_space["agent"])
        self.target_model = build_model(action_space.n,obs_space["agent"])
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # with probability epsilon return a random action to explore the environment
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space.n)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        states = np.concatenate(minibatch[:, 0])
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2]
        next_states = np.concatenate(minibatch[:, 3])
        dones = minibatch[:, 4]
        q_values = self.model.predict(states)
        q_next = self.target_model.predict(next_states)
        q_targets = rewards + (1 - dones) * self.gamma * np.amax(q_next, axis=1)
        q_values[np.arange(len(states)), actions] = q_targets
        self.model.fit(states, q_values, epochs=1, verbose=0)


