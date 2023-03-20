import os
import tempfile
def boundingBox(picture, save):

    import cv2

    image = cv2.imread(picture)
    predictions = {'predictions': [{'x': 130.0, 'y': 100.5, 'width': 170.0, 'height': 100.0, 'confidence': 0.7369905710220337, 'class': 'Paper', 'image_path': 'example.jpg', 'prediction_type': 'ObjectDetectionModel'}], 'image': {'width': 1436, 'height': 956}}

    for bounding_box in predictions["predictions"]:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(image, start_point, end_point, color=(255,0,0), thickness=2)
    
    cv2.imwrite(save, image)

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_cv_mode")

count = 0
for root_dir, cur_dir, files in os.walk(tmp_dir):
    count += len(files)

for interaction in range(count):
    filename = os.path.join(tmp_dir,'image' + "_" + str(interaction)+".png")
    print(filename)
    saveDir = os.path.join("/home/caio/Desktop/test/",'imageBB' + "_" + str(interaction)+".png")
    boundingBox(filename, saveDir)