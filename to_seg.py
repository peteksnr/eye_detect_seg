import json
import numpy as np
import cv2
from pycocotools import mask
from PIL import Image

class_mapping = {
    "iris": 36, 
    "pupil": 12  
}

with open("/Users/peteksener/Downloads/iris_Seg-2/train/_annotations.coco.json", "r") as f:
    coco_data = json.load(f)

for image_info in coco_data["images"]:
    image_id = image_info["id"]
    width, height = image_info["width"], image_info["height"]
    file_name = str(image_info['file_name']).replace('jpg', 'png')
    
    mask_img = np.zeros((height, width), dtype=np.uint8)
    
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
    
    for ann in annotations:
        segmentation = ann["segmentation"]
        category_id = ann["category_id"]

        class_name = next(cat["name"] for cat in coco_data["categories"] if cat["id"] == category_id)

        pixel_value = class_mapping.get(class_name, 0) 

        for segment in segmentation:
            pts = np.array(segment, np.int32).reshape((-1, 2))
            cv2.fillPoly(mask_img, [pts], pixel_value)
    
    mask_path = f"/Users/peteksener/Downloads/iris_Seg-2/train_mask/{file_name}"
    Image.fromarray(mask_img).save(mask_path)
    print(f"Saved mask: {mask_path}")