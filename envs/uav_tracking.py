import numpy as np
import airsim
import gymnasium as gym
from gymnasium import spaces

class TrackingEnv(gym.Env):

    def __init__(self, discrete_actions = True,ep_lenght=50):
        
        self._state = 0
        self.ep_lenght = ep_lenght
        self.uav = airsim.MultirotorClient()
        self.map_size = 5e2

        # Adjust to possible possitions of agent and target
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-self.map_size, -self.map_size, shape=(2,), dtype=float),
                "target": spaces.Box(-self.map_size, -self.map_size, shape=(2,), dtype=float),
            })
        if discrete_actions:
            # Up, down, front, back, left and rigth
            self.action_space = spaces.Discrete(6)
        else:
            # Aceleration or NED - TO-DO
            self.action_space = spaces.Box(0, 1, (4,))
    
    def _get_obs(self):
        pass

    def render(active= False):
        #Check how disable render ate unreal
        pass

    def step(self,u):
        reward = 0
        done = self._state >= self.ep_lenght
        observation = self._get_obs()
        state = [0,0,0,0,0,0] #Discrete
        info = state
        return observation, reward, done, info

    def reset (self, seed = None, options = None):
        super().reset(seed=seed)
        return 0
    
    def close(self):
        #Land the drone and finish the simulation
        pass