import gymnasium as gym
from envs.uav_tracking import TrackingEnv
from dqn import DQNAgent

# Instantiate your custom environment
env = TrackingEnv()

# Define hyperparameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Instantiate DQNAgent
agent = DQNAgent(env.action_space, env.observation_space)

# Train the agent
num_episodes = env.ep_lenght
for episode in range(num_episodes):
    state = env.reset()
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
