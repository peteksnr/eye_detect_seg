import os
import cv2

input_dir = "/Users/peteksener/Desktop/eyes"
output_dir = "/Users/peteksener/Desktop/w"

os.makedirs(output_dir, exist_ok=True)

def read_yolo_annotations(anno_path, img_width, img_height):
    boxes = []
    with open(anno_path, "r") as file:
        for line in file.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue  
            _, x_center, y_center, width, height = map(float, parts)

            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            boxes.append((x1, y1, x2, y2))
    return boxes

for file_name in os.listdir(input_dir):
    if file_name.endswith((".jpg", ".png")): 
        img_path = os.path.join(input_dir, file_name)
        anno_path = os.path.join('/Users/peteksener/Desktop/s2/eyedetect/images', file_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        if not os.path.exists(anno_path):
            print(f"Warning: Annotation file missing for {file_name}")
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Failed to read {file_name}")
            continue

        img_h, img_w, _ = image.shape
        boxes = read_yolo_annotations(anno_path, img_w, img_h)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cropped_eye = image[y1:y2, x1:x2]
            eye_output_path = os.path.join(output_dir, f"{file_name[:-4]}_eye{i+1}.jpg")
            cv2.imwrite(eye_output_path, cropped_eye)

print("Processing complete.")