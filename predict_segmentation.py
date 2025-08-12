import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

class Predict_Segmentation():
    def __init__(self, model_path):
        self.model = smp.Unet(
            encoder_name='resnet34', 
            encoder_weights=None, 
            in_channels=3, 
            classes=1,  
        )
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')), strict=False)
        self.model.eval() 


    def preprocess_image(self, image_path, target_size=(256, 256)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image_resized = cv2.resize(image, target_size) 

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_resized).unsqueeze(0) 
        return image, image_tensor 


    def predict_mask(self, model, image_tensor):
        with torch.no_grad():
            output = model(image_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy() 
            return mask > 0.5  


    def visualize_segmentation(self,masked_img, original_image, mask):
        mask_resized = cv2.resize(mask.astype(np.uint8) * 255, (original_image.shape[1], original_image.shape[0]))

        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(masked_img)
        plt.axis("off")
        plt.title("original")
        
        plt.subplot(1, 2, 2)
        plt.imshow(original_image)
        plt.imshow(mask_resized, cmap="hot", alpha=0.5) 
        plt.axis("off")
        plt.title("predicted")
        plt.show()

    def visualize_segment(self, masked_img, original_image, mask, loss, f1, iou):
        mask_binary = (mask > 0).astype(np.uint8) * 255 
        mask_resized = cv2.resize(mask_binary, (original_image.shape[1], original_image.shape[0]))
        black_background = np.zeros_like(mask_resized)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.axis("off")
        plt.title("original")

        plt.subplot(1, 3, 2)
        plt.imshow(masked_img)
        plt.axis("off")
        plt.title("ground truth mask")

        plt.subplot(1, 3, 3)
        plt.imshow(black_background, cmap="gray")  
        plt.imshow(mask_resized, cmap="gray", alpha=1.0)
        plt.axis("off")
        plt.title("predicted")

        text_x = original_image.shape[1] * 0.01 
        text_y = original_image.shape[0] * 0.1  

        plt.text(
            text_x, text_y, 
            f"loss: {loss:.4f}\nf1: {f1:.4f}\niou: {iou:.4f}", 
            fontsize=12, color="white", bbox=dict(facecolor="black", alpha=0.5)
        )

        plt.show()