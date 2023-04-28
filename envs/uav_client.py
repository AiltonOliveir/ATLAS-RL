import numpy as np
import airsim

class UAV():
    def __init__(self):
        print('Connecting Airsim')
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print('Airsim already')
        self.image_request = airsim.ImageRequest(0, airsim.ImageType.Scene) #png
        self.encode_action = {
            (0,0,1): 0,(0,0,-1): 1,
            (1,0,0): 2, (1,1,0): 3,
            (0,1,0): 4, (-1,1,0):5,
            (-1,0,0):6,(-1,-1,0):7,
            (0,-1,0):8, (1,-1,0): 9
            }
    
    def _start_fligth(self,uav_id="uav1"):
        self.client.enableApiControl(True, uav_id)
        self.client.armDisarm(True, uav_id)
        self.client.takeoffAsync(vehicle_name=uav_id).join()

    def _gps_get(self,uav_id="uav1"):
        """
        The postition inside the returned MultiRotorState is in the frame of the vehicle's starting point
        
        parameters: vehicle_name(str,optional) vehicle to get the state of

        returns:

        can_arm=False
        collision=<collisioninfo>{}
        gps_location=<GeoPoint>{}
        kinematics_estimated=<KinematicState>{}
        landed_state=0
        rc_data={}
        ready= False
        ready_message="
        timestamp=0

        """
        x = self.client.getMultirotorState(vehicle_name=uav_id)
        return x

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

if __name__== "__main__":
    x = UAV()
    gps_data = x._gps_get()

    print(type(gps_data))
    print(gps_data)
