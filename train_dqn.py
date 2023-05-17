import gymnasium as gym
import tensorflow as tf
from envs.uav_tracking import TrackingEnv
from dqn import DQNAgent

# Instantiate your custom environment
env = TrackingEnv()
# Instantiate DQNAgent
agent = DQNAgent(env.action_space, env.observation_space)

# Train the agent
num_episodes = env.ep_lenght
state = env.reset()
for episode in range(num_episodes):
    done = False
    # Choose an action using epsilon-greedy policy
    action = agent.act(state)
    # Take a step in the environment
    next_state, reward, done, _ = env.step(action)
    # Add the experience to the agent's replay buffer
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    # Update the agent's Q-network and target Q-network
    agent.replay()
    if episode % 10 == 0:
        agent.target_train()
    # Decay epsilon
    agent.update_epsilon()
    print("Episode: {}, Total reward: {}, Epsilon: {:.2f}".format(episode, env.reward, agent.epsilon))
