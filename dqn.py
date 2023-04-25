import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

# Build a neural network for your DQN agent
"""This method, creates a neural network with 2 hidden layers of 64 neurons each and an output layer 
with the number of neurons equal to the number of actions available in the environment. 
The activation function used in the hidden layers is ReLU, 
and the output layer has no activation function (linear activation). 
The loss function used is mean squared error, and the optimizer used is 
Adam(uses estimations of the first and second moments of the gradient to adapt the learning rate for each weight 
of the neural network.) with a learning rate of 0.001."""
def build_model(action_space,obs_space):
    # Build and compile Q-network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(obs_space.shape)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Define the DQN agent
"""The DQNAgent class defines the agent itself. It has several attributes, 
including the epsilon-greedy exploration probability (epsilon), 
the minimum exploration probability (epsilon_min), 
the exploration probability decay rate (epsilon_decay), 
the discount factor (gamma), the batch size (batch_size), 
the action space (action_space), the memory buffer (memory), 
the Q-network model (model), and the target Q-network model (target_model)."""
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
        """The act method chooses an action to take given the current state. 
    With probability epsilon, it chooses a random action to explore the environment, 
    otherwise, it chooses the action with the highest Q-value predicted by the Q-network."""
        # with probability epsilon return a random action to explore the environment
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space.n)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """The remember method adds the current state, 
        action taken, 
        reward received, next state, 
        and whether the episode is finished or not to the memory buffer."""
        self.memory.append((state, action, reward, next_state, done))
   
    def update_epsilon(self):
        """The update_epsilon method updates the exploration probability based on the decay rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        """The target_train method updates the target Q-network model 
        by copying the weights from the Q-network model."""
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        """The replay method performs the training of the Q-network model. 
        It first samples a batch of experiences from the memory buffer. 
        Then, it computes the Q-values for the current states and the Q-values
        for the next states using the Q-network model and the target Q-network model, respectively. 
        It updates the Q-values for the current states using the Q-values for the next states 
        and the rewards received. Finally, it fits the Q-network model using the updated Q-values 
        for the current states."""
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
