import airsim
import os
import tempfile
import numpy as np
import random as rd
import json

#Correction of a bug that input twice
def cBug1():
    airsim.wait_key()

#Defining the name and type of camera
CAM_NAME = "fixed1"
IS_EXTERNAL_CAM = True

#start some global variables
typePose= 0
interaction = 0

listCarPosition = []
listCameraPosition = []
listFov = []

#starting the conection with airsim
client = airsim.VehicleClient()
client.confirmConnection()

#Making the path to store the pictures
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_cv_mode")
print ("Saving images t %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

#starting the simulation
airsim.wait_key('Press any key to get images')
cBug1()

#defining what type of image we want
requests = [airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene)]

#starting takng pictures
while interaction < 1000:
    print(interaction)
    #getting the pose of the car and spliting in position and orientation
    objectPose = client.simGetObjectPose("carro")
    objectPosition = objectPose.position
    objectOrientation = objectPose.orientation

    #correction factor of car's z_val
    objectZajust = objectPosition.z_val - 1.012

    #Puting the camera close to the car
    
    x = objectPosition.x_val + 3
    y = objectPosition.y_val 

    #Defining the camera's position
    a = 2 + (2*rd.random())
    positions = [(x, y, 0), (x, y + a, 0), (x-a, y+6, 0), (x -6 -a, y+6, 0), (x - 12, y + a, 0), (x - 12, y - a, 0), (x -6 -a, y-6, 0), (x, y-a, 0),
                 (x, y, -a), (x, y + a, -a), (x-a, y+6, -a), (x -6 -a, y+6, -a), (x - 12, y + a, -a), (x - 12, y - a, -a), (x -6 -a, y-6, -a), (x, y-a, -a)]

    filename = os.path.join(tmp_dir,'image' + "_" + str(interaction))
    
    auxAngYaw = np.arange(-0.6, 0.6, 0.001)
    auxAngYawInd = rd.randint(0, len(auxAngYaw)-1)

    auxAngPitch = np.arange(-0.1, 0.1, 0.01)
    auxAngIndPitch = rd.randint(0, len(auxAngPitch)-1)
    #getting the relative distance between camera and car
    x_rel = positions[typePose][0]-objectPosition.x_val
    y_rel = positions[typePose][1]-objectPosition.y_val
    z_rel = positions[typePose][2]-(objectZajust)

    #Adjusting the camera pitch if it is higher than the car
    if z_rel < -2:
        pitch = -np.arctan(-z_rel/np.sqrt((x_rel**2)+(y_rel)**2))
        
    else:
         pitch = 0

    #Adjusting the camera yaw according to quadrant     
    if x_rel >= 0 and y_rel >= 0:
        yaw = np.arctan((y_rel)/(x_rel))
        yaw = np.pi + yaw

    elif x_rel < 0 and y_rel > 0:
        yaw = np.arctan((y_rel)/(-x_rel))
        yaw = -yaw
    
    elif x_rel < 0 and y_rel < 0:
        yaw = np.arctan((-y_rel)/(-x_rel))

    elif x_rel > 0 and y_rel < 0:
         yaw = np.arctan((-y_rel)/(x_rel))
         yaw = np.pi-yaw

    else:
         yaw = 0
    
    shouldSeeyaw = yaw
    shouldSeeyawpitch = pitch
    yaw = yaw + auxAngYaw[auxAngYawInd]
    pitch = pitch + auxAngPitch[auxAngIndPitch]
    #distance = np.sqrt((x_rel**2)+(y_rel**2)+(z_rel**2))
    #nfov = 270/distance
    #client.simSetCameraFov("fixed1",nfov,external=True)
    fov = client.simGetCameraInfo('fixed1', external=True)
    
 
    dictInformation = { "car_position" : [objectPosition.x_val, objectPosition.y_val, objectPosition.z_val],
                        "camera_position": positions[typePose] , "camera_fov" : fov.fov, "diffYaw" :auxAngYaw[auxAngYawInd], 
                        "diffPitch": auxAngPitch[auxAngIndPitch]}
    

    jsonString = json.dumps(dictInformation, indent=7)

    with open("/home/caio/Documents/Drone/Codes/dataJson/data"+str(interaction)+".json", "w") as jsonFile:
    #start procedure
        jsonFile.write(jsonString)
        jsonFile.close

    #setting the new camera's pose
    npose = airsim.Pose(position_val=airsim.Vector3r(positions[typePose][0], positions[typePose][1], positions[typePose][2] ), orientation_val=airsim.to_quaternion(pitch, 0, yaw ))
    client.simSetCameraPose(camera_name=CAM_NAME, pose=npose, external=IS_EXTERNAL_CAM)
    
    #getting the pictures
    responses = client.simGetImages(requests, external=IS_EXTERNAL_CAM)
    for i, response in enumerate(responses):
        if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        else:
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    
    #adjust to the next type of camera's position
    typePose = typePose+1
    if typePose >= 16:
         typePose = 0
    
    interaction = interaction + 1

#Angulos estao em radiano