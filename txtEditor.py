import os
import numpy as np
import tempfile
import json
import cv2

count = 0
for root_dir, cur_dir, files in os.walk("/home/caio/Documents/Drone/Codes/dataJson"):
    count += len(files)

for i in range(count):
    file = "/home/caio/Documents/Drone/Codes/dataJson/data" + str(i) + ".json"
    a = open(file, 'r')
    ajson = json.load(a)

    x_rel = ajson["car_position"][0] - ajson["camera_position"][0]
    y_rel = ajson["car_position"][1] - ajson["camera_position"][1]
    z_rel = ajson["car_position"][2] - ajson["camera_position"][2]

    distanceCC = np.sqrt((x_rel**2) + (y_rel**2)
                         + (z_rel**2))
    angYaw = ajson["diffYaw"]
    angPitch = ajson["diffPitch"]
    midX = 130
    midY = 80
    correctBB_X = 15*np.sqrt(2*(distanceCC**2) - (2*(distanceCC**2)*np.cos(np.abs(angYaw))))
    correctBB_Y = 15*np.sqrt(2*(distanceCC**2) - (2*(distanceCC**2)*np.cos(np.abs(angPitch))))

    if angYaw < 0:
        midX = midX + correctBB_X
    else:
        midX = midX - correctBB_X

    if angPitch < 0:
        midY = midY - correctBB_Y
    else:
        midY = midY + correctBB_Y

    
    tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_cv_mode")
    picture = os.path.join(tmp_dir,'image' + "_" + str(i)+".png")
    saveDir = os.path.join("/home/caio/Desktop/test/",'imageBB' + "_" + str(i)+".png")
    
    image = cv2.imread(picture)

    predictions = {'predictions': [{'x': midX, 'y': midY, 'width': 80.0, 'height': 50.0, 
                                    'confidence': 0.7369905710220337, 'class': 'Paper', 'image_path': 'example.jpg', 'prediction_type': 'ObjectDetectionModel'}], 'image': {'width': 1436, 'height': 956}}
    for bounding_box in predictions["predictions"]:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(image, start_point, end_point, color=(255,0,0), thickness=2)
    
    cv2.imwrite(saveDir, image)