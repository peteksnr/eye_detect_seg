import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.utils.data import DataLoader
from load_data import Load_Data
import numpy as np
from sklearn.metrics import f1_score
import torch

data = Load_Data()

train_images_dir = "/Users/peteksener/Desktop/t/img"
train_masks_dir = "/Users/peteksener/Desktop/t/mask"
val_images = "/Users/peteksener/Desktop/v/img"
val_masks = "/Users/peteksener/Desktop/v/mask"

train_dataset = data.get_dataset(train_images_dir, train_masks_dir, transform=data.image_transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = data.get_dataset(val_images, val_masks, transform=data.image_transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

#resnet152 densenet161 efficientnet-b3 inceptionresnetv2
model = smp.Unet(
    encoder="resnet152",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load('/Users/peteksener/Desktop/y/iris_resnet34.pth', map_location=torch.device('mps')), strict=False)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total_pixels = 0
    
    for image, mask in train_dataloader:
        image, mask = image.to(device), mask.to(device)

        optimizer.zero_grad()
        result = model(image)
        
        loss = criterion(result, mask)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        predicted = torch.sigmoid(result) > 0.5 
        correct += (predicted == mask).sum().item() 
        total_pixels += mask.numel()
        

    model.eval()
    validation_loss = 0.0
    correct_val = 0
    total_val_pixels = 0
    iou_scores = []
    f1_scores = []

    with torch.no_grad():
        for img, masks in val_dataloader:
            img, masks = img.to(device), masks.to(device)
            result = model(img)
            
            loss = criterion(result, masks)
            validation_loss += loss.item() * img.size(0)
            
            predicted = torch.sigmoid(result) > 0.5
            correct_val += (predicted == masks).sum().item()
            total_val_pixels += masks.numel()

            masks = masks > 0.5  

            intersection = (predicted * masks).float().sum().item()
            union = (predicted + masks).float().sum().item()
            iou = intersection / (union + 1e-8)
            iou_scores.append(iou)

            f1 = f1_score(masks.cpu().numpy().flatten(), predicted.cpu().numpy().flatten(), zero_division=1)
            f1_scores.append(f1)

    val_accuracy = correct_val / total_val_pixels
    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)

    accuracy = correct / total_pixels 

    print(f"Epoch {epoch+1}, train loss: {epoch_loss/len(train_dataloader)}, train accuracy: {accuracy}\n" 
                             f"val loss: {validation_loss/len(val_dataloader)}, val accuracy: {val_accuracy}, iou: {avg_iou}, f1 score: {avg_f1}")

torch.save(model.state_dict(), "iris_seg_pre.pth")