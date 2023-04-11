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
            self.action_space = 

            """
            moveByVelocityAsync(vx, vy, vz, duration=0, yaw_mode=<YawMode>{ 'is_rate':True, 'yaw_or_rate':0.0}, vehicle_name=")

            Parameters:

            vx(float): Desired velocity in world (NED) X axis
            vy(float): Desired velocity in world (NED) Y axis
            vz(float): Desired velocity in world (NED) Z axis
            duration: Desired amount of time(seconds), to send this command for
            drivetrain(DrivetrainType, optional)-
            yaw_mode(YawMode, optional) - 
            vehicle_name(str, optional) - Name of the multirotor to send this command to 

            Returns:

            client.METHOD().join()  
            """

    
    def _get_obs(self):
        pass

    def render(active= False):
        #Check how disable render ate unreal
        pass

    def _compute_reward(self):
        """
        To develop a simple reward mechanic for the drone to go from point A to point B, 
        We define a reward based on the distance traveled by the drone in the direction of the target,
        calculating the distance between the current position of the drone and the target position, 
        and then subtract the distance in the next time step from the current distance. 
        This difference would give us the progress the drone made towards the target, and we can use this as the reward.
        """

        drone_pos = self._get_obs()["agent"]
        target_pos = self._get_obs()["target"]
        distance_to_target = np.linalg.norm(drone_pos - target_pos)

        # Move the drone in the direction of the target and measure the progress
        self.uav.discrete_action(2)  # Move North
        new_drone_pos = self._get_obs()["agent"]
        new_distance_to_target = np.linalg.norm(new_drone_pos - target_pos)
        distance_moved_towards_target = distance_to_target - new_distance_to_target

        reward = distance_moved_towards_target
        return reward


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