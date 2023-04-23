import numpy as np
import airsim
from .uav_client import UAV
import gymnasium as gym
from gymnasium import spaces

class TrackingEnv(gym.Env):

    def __init__(self, discrete_actions = True,ep_lenght=100, goal = [20,20,10]):
        
        self._state = 0
        self.ep_lenght = ep_lenght
        self.uav = UAV()
        self.map_size = 5e2
        self.max_distance = 100 # maximum distance between agent and target
        self.threshold = 10 # threshold distance for giving reward
        self.max_episode_steps = 100 # variable to define the maximum number of steps in an episode.
        self.current_step = 0 # variable to keep track of the current step in the episode
        self.done = False # variable to keep track of whether the episode is done or not.
        self.reward = 0.0 # parameter to store the cumulative reward
        self.goal_position = np.array(goal)

        # Adjust to possible positions of agent and target
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-self.map_size, self.map_size, shape=(3,), dtype=float),
                "target": spaces.Box(-self.map_size, self.map_size, shape=(3,), dtype=float),
            })
        if discrete_actions:
            # Down(Z), Up(-Z), North(X), North-east, East(Y), South-East, South(-X), South-West, Weast(-Y) and North-West
            self.action_space = spaces.Discrete(10)
        else:
            # Aceleration or NED - TO-DO
            self.action_space = spaces.Box(0, 1, (4,))

    
    def _get_obs(self):
        """
        Return the current observation of the environment. 
        """
        # Get the current state of the drone
        state = self.uav.client.getMultirotorState()
        # Get the position and orientation of the drone
        achieved_goal = np.array([state.kinematics_estimated.position.x_val,
                            state.kinematics_estimated.position.y_val,
                            state.kinematics_estimated.position.z_val])

        '''orientation = np.array([state.kinematics_estimated.orientation.w_val,
                                state.kinematics_estimated.orientation.x_val,
                                state.kinematics_estimated.orientation.y_val,
                                state.kinematics_estimated.orientation.z_val])'''
        return achieved_goal
    
    def render(active= False):
        #Check how disable render ate unreal
        pass

    def _compute_reward(self,achieved_goal, desired_goal, info):
        """
        To develop a simple reward mechanic for the drone to go from point A to point B, 
        Computes the reward for a given achieved goal and the desired goal.
        Penalizes the agent if it moves away from the goal compared to the previous step.
        """
        self.reward += self._compute_reward(achieved_goal, desired_goal, info)
        # Compute the distance to the goal for the current and previous steps        
        distance_current = np.linalg.norm(achieved_goal - desired_goal)
        distance_previous = np.linalg.norm(info['achieved_goal'] - desired_goal) if 'achieved_goal' in info else distance_current

        # Check if the agent has reached the goal position
        if distance_current < self.threshold:
            reward = 1.0
        else:
            # Compute a reward based on the distance to the goal
            if distance_current < self.max_distance:
                reward = (self.max_distance - distance_current) / self.max_distance
            else:
                reward = 0.0

        # Penalize the agent if it moves away from the goal
        if distance_current > distance_previous:
            reward -= 0.5

        return reward

    def step(self,action):
        """
        The reward is calculated based on the agent's position relative to the goal position. As the agent gets closer to the goal position, the reward increases. 
        If the agent reaches the goal position, the reward is set to 1. 
        To make sure the agent cannot move backward after reaching the goal position, 
        we check if the current position is equal to the goal position and the action is to move backward. 
        If so, we set the done flag to True, indicating the end of the episode and the reward to -1 if the agent tries to move backward after reaching the goal position. This is to discourage the agent from attempting to move backward once it has reached the goal position
        """
        self.uav.discrete_action(action)
        observation = self._get_obs()
        done = self._state >= self.ep_lenght
        info = {'achieved_goal': observation, 'desired_goal': self.goal_position, 'action': action}

        # Move the agent according to the action
        if action == 0: # move forward
            self.position += 1
        elif action == 1: # move backward
            self.position -= 1

        # Update the reward based on the new position
        reward = self._compute_reward(observation, self.goal_position, info)

        # Check if the episode is done
        if self.position == self.goal_position and action == 1: # reached goal and tries to move backward
            reward = -1
            done = True
        elif self.position == self.goal_position: # reached goal
            reward = 1
            done = True

        return observation, reward, done, info

    def reset (self, seed = None, options = None):
        super().reset(seed=seed)
        self.uav._reset_flight()
        return 0
    
    def close(self):
        #Land the drone and finish the simulation
        self.uav.airsim_land()
        #To-do (Finish the simulation method)