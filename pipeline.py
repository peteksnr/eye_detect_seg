from ultralytics import YOLO
import cv2 as cv
from predict_segmentation import Predict_Segmentation
import os
from PIL import Image, ImageEnhance
import numpy as np

def detect(image, model):
    model = YOLO(model)
    results = model(image)
    imigi = cv.imread(image)
    detections = []
    for i, result in enumerate(results):
        boxes = result.boxes 
        for j, box in enumerate(boxes.xyxy): 
            x1, y1, x2, y2 = map(int, box.tolist())
            eye = imigi[y1:y2, x1:x2]
            image_name = str(os.path.basename(image)).replace('.png','')
            cv.imwrite(f'/path/to/eye/images/{image_name}_eye_{j}.png', eye) # replace with your path
            detections.append(f'/path/to/eye/images/{image_name}_eye_{j}.png') # replace with your path
        result.show() 
    return detections
        
def segment(image_path, model):
    original_image = cv.imread(image_path)
    if image_path == None:
        print('No detected eyes are found')
    predict = Predict_Segmentation(model_path=model)
    image, tensor = predict.preprocess_image(image_path)
    mask_pred = predict.predict_mask(predict.model, tensor)
    predict.visualize_segmentation(original_image, image, mask_pred)


def pipeline(img, detect_model, segment_model):
    eyes = detect(img, detect_model)
    for eye in eyes:
        segment(eye, segment_model)


pipeline(img='/path/to/image', 
         detect_model='/path/to/detection/model', 
         segment_model='/path/to/segmentation/model')


        
