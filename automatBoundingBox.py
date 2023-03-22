import os
import numpy as np
import tempfile
import json
import cv2

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

yolobox = []
count = 0
#counting the numbers of photos
for root_dir, cur_dir, files in os.walk("/home/caio/Documents/Drone/Codes/dataJson"):
    count += len(files)

#Generating the bounding box for each photo 
for i in range(count):
    #getting the photo's information
    file = "/home/caio/Documents/Drone/Codes/dataJson/data" + str(i) + ".json"
    a = open(file, 'r')
    ajson = json.load(a)

    #caculating the distance between car and camera
    x_rel = ajson["car_position"][0] - ajson["camera_position"][0]
    y_rel = ajson["car_position"][1] - ajson["camera_position"][1]
    z_rel = ajson["car_position"][2] - ajson["camera_position"][2]
    distanceCC = np.sqrt((x_rel**2) + (y_rel**2)
                         + (z_rel**2))
    #Getting the angular "error" of the camera's rotation
    angYaw = ajson["diffYaw"]
    angPitch = ajson["diffPitch"]

    #deffining a central point for the bounding box
    midX = 130
    midY = 80

    #deffining a proportional width and heigh using the distance Camera-car as base
    width = 80 * 6.122 / distanceCC 
    heigh = 50 *6.122/ distanceCC

    #Calculating the correction factor for the bounding box centralize the car 
    if distanceCC > 8:
        correctBB_X = (23*6.122/distanceCC)*np.sqrt(2*(distanceCC**2) - (2*(distanceCC**2)*np.cos(np.abs(angYaw))))
    else:
        correctBB_X = (18*6.122/distanceCC)*np.sqrt(2*(distanceCC**2) - (2*(distanceCC**2)*np.cos(np.abs(angYaw))))
    correctBB_Y = 12*np.sqrt(2*(distanceCC**2) - (2*(distanceCC**2)*np.cos(np.abs(angPitch))))

    #analysing if the car is more left or more right in the photo
    if angYaw < 0:
        midX = midX + correctBB_X
    else:
        midX = midX - correctBB_X

    #analysing if the car is upper or bottom in the photo
    if angPitch < 0:
        midY = midY - correctBB_Y
    else:
        midY = midY + correctBB_Y

    #generating the photos with the bounding box
    tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_cv_mode")
    picture = os.path.join(tmp_dir,'image' + "_" + str(i)+".png")
    saveDir = os.path.join("/home/caio/Desktop/test/",'imageBB' + "_" + str(i)+".png")
    
    image = cv2.imread(picture)

    predictions = {'predictions': [{'x': midX, 'y': midY, 'width': width, 'height': heigh, 
                                    'confidence': 0.7369905710220337, 'class': 'Paper', 'image_path': 'example.jpg', 'prediction_type': 'ObjectDetectionModel'}], 'image': {'width': 1436, 'height': 956}}
    aux = 0
    for bounding_box in predictions["predictions"]:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2

        im=image.shape
        w= int(im[1])
        h= int(im[0])
        b = (x0, x1, y0, y1)
        bb = convert((w,h), b)
        yolobox.append(bb)

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(image, start_point, end_point, color=(255,0,0), thickness=2)

        yolofile = open("/home/caio/Documents/Drone/Codes/yoloData/imageBB" + "_" + str(i) + ".txt", 'w+')
        txt =str(1) + " " + str(yolobox[i][0]) + " " + str(yolobox[i][1]) + " " + str(yolobox[i][2]) + " " + str(yolobox[i][3])
        yolofile.write(txt)
        yolofile.close()

    cv2.imwrite(saveDir, image)
