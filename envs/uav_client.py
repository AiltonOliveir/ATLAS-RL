import numpy as np
import airsim

class UAV():
    def __init__(self):

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.image_request = airsim.ImageRequest(0, airsim.ImageType.Scene) #png
        self.encode_action = {
            [0,0,1]: 0,[0,0,-1]: 1,
            [1,0,0]: 2, [1,1,0]: 3,
            [0,1,0]: 4, [-1,1,0]:5,
            [-1,0,0]:6,[-1,-1,0]:7,
            [0,-1,0]:8, [1,-1,0]: 9
            }
    
    def _start_fligth(self,uav_id="uav1"):
        self.client.enableApiControl(True, uav_id)
        self.client.armDisarm(True, uav_id)
        self.client.takeoffAsync(vehicle_name=uav_id).join()
    
    def _reset_flight(self):
        self.client.reset()
        self._start_fligth()
    
    def airsim_land(self,uav_id="uav1"):
        landed = self.client.getMultirotorState(vehicle_name=uav_id).landed_state
        #Ailton OBS: Check if this if works
        if landed == airsim.LandedState.Landed:
            print("already landed...")
            return None
        self.client.armDisarm(False, uav_id)

    def discrete_action(self,agent_action,speed = 5,uav_id="uav1"):
        x,y,z = self.encode_action[agent_action]
        self.client.enableApiControl(True)
        self.client.moveToPositionAsync(x, y, z, speed, 1e+38, airsim.DrivetrainType.ForwardOnly,
                                         airsim.YawMode(False,0), vehicle_name=uav_id)

    def _positions(self):
        pass
