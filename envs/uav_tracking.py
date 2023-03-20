import numpy as np
import airsim
from uav_client import UAV
import gymnasium as gym
from gymnasium import spaces

class TrackingEnv(gym.Env):

    def __init__(self, discrete_actions = True,ep_lenght=50):
        
        self._state = 0
        self.ep_lenght = ep_lenght
        self.uav = UAV()
        self.map_size = 5e2

        # Adjust to possible possitions of agent and target
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
        pass

    def render(active= False):
        #Check how disable render ate unreal
        pass

    def _compute_reward(self):
        pass

    def step(self,action):
        self.uav.discrete_action(action)
        observation = self._get_obs()
        reward = self._compute_reward()
        done = self._state >= self.ep_lenght
        state = [0,0,0,0,0,0] #Discrete
        info = state
        return observation, reward, done, info

    def reset (self, seed = None, options = None):
        super().reset(seed=seed)
        self.uav._reset_flight()
        return 0
    
    def close(self):
        #Land the drone and finish the simulation
        self.uav.airsim_land()
        #To-do (Finish the simulation method)