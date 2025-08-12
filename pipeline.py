from ultralytics import YOLO
import cv2 as cv
from predict_segmentation import Predict_Segmentation
import os
from PIL import Image, ImageEnhance
import numpy as np

def detect(image, model):
    # '/Users/peteksener/Desktop/y/runs/detect/train11/weights/best.pt'
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
            cv.imwrite(f'/Users/peteksener/Desktop/y/eyes/{image_name}_eye_{j}.png', eye)
            detections.append(f'/Users/peteksener/Desktop/y/eyes/{image_name}_eye_{j}.png')
        result.show() 
    return detections
        
def segment(image_path, model):
    # '/Users/peteksener/Desktop/y/iris_seg_pre.pth'
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


pipeline(img='/Users/peteksener/Downloads/ben.jpeg', 
         detect_model='/Users/peteksener/Desktop/y/runs/detect/train11/weights/best.pt', 
         segment_model='/Users/peteksener/Desktop/y/iris_seg_pre.pth')


        