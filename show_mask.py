import matplotlib.pyplot as plt
import cv2
from predict_segmentation import Predict_Segmentation
import numpy as np
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import torch


image_path = "/Users/peteksener/Desktop/t/img/010_07_eye1_jpg.rf.4f1ae532c317067656c7fc14c1107079__4.jpg"
mask_path = "/Users/peteksener/Desktop/t/mask/010_07_eye1_jpg.rf.4f1ae532c317067656c7fc14c1107079__4.png"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
mask_tensor_filtered = torch.where(mask_tensor == 36, torch.tensor(36.0), torch.tensor(0.0))
print(torch.unique(mask_tensor_filtered))

color_mask = np.zeros_like(image)
color_map = {36: [255, 255, 255]}
for class_id, color in color_map.items():
    color_mask[mask == class_id] = color


predict = Predict_Segmentation(model_path='/Users/peteksener/Desktop/y/iris_seg_pre.pth')
image, tensor = predict.preprocess_image(image_path)
mask_pred = predict.predict_mask(predict.model, tensor)




def calculate_metrics(img, masks, model):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        img, masks = img.to(device), masks.to(device)
        masks = (masks > 0).float()

        masks_resized = F.interpolate(masks, size=(256, 256), mode="nearest")

        result = model(img)
        loss = criterion(result, masks_resized).item()

        predicted = torch.sigmoid(result) > 0.5
        masks_resized = masks_resized > 0.5  

        predicted = predicted.int()
        masks_resized = masks_resized.int()

        intersection = (predicted & masks_resized).float().sum().item()
        union = (predicted | masks_resized).float().sum().item()
        iou = intersection / (union + 1e-8)  

        f1 = f1_score(
            masks_resized.cpu().numpy().flatten(), 
            predicted.cpu().numpy().flatten(), 
            zero_division=1
        )

        print(f'loss: {loss}, f1: {f1}, iou: {iou}')

        return loss, f1, iou

loss, f1, iou = calculate_metrics(tensor, mask_tensor, predict.model)
# predict.visualize_segmentation(overlay, image, mask_pred)
predict.visualize_segment(color_mask, image, mask_pred, loss, f1, iou)
