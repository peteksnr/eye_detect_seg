# 👁️ Eye Detection & Iris Segmentation Pipeline

This project implements a **two-stage deep learning pipeline** for detecting the eye region in an image and performing iris segmentation.  
It combines **YOLOv11n** 🦾 object detection with **U-Net** 🧩 semantic segmentation using various encoder backbones.

---

## 📌 Project Overview
The pipeline follows these main steps:

1. **🔍 Eye Detection**  
   - A custom-trained YOLOv11n model detects the eye region in the input image.  
   - The detected bounding box is used to crop the eye area.

2. **🛠 Data Preparation for Segmentation**  
   - Cropped eye images are prepared for segmentation training.  
   - Augmentation techniques are applied using **Albumentations** 🎨 and **imgaug** to improve robustness.

3. **🎯 Iris Segmentation**  
   - The cropped eye image is passed to a U-Net segmentation model with various encoders (`ResNest152`, `DenseNet161`, `InceptionResNetV2`, `EfficientNet-b3`).  
   - The predicted mask is generated and visualized.

4. **⚙️ Pipeline Integration**  
   - All steps are combined into a single `pipeline()` function that runs detection → cropping → segmentation → visualization.

---

## 📂 Dataset Preparation
- **📷 Detection Dataset**: Labeled in YOLO format, verified with **LabelImg**.  
- **🎭 Segmentation Dataset**: Eye crops masked using **Roboflow** for iris segmentation.  
- **🎨 Augmentation**: Applied using **Albumentations** and `imgaug` to simulate variations in lighting, rotation, and quality.  
- **🔄 Pretrained Fine-tuning**: YOLO model retrained starting from a previously trained model for improved performance.

---

## 📊 Model Performance Summary
| 🧠 Model             | 📈 Train Acc. | 📊 Val Acc. | 📏 IoU   | 🏆 Best Metric      |
|----------------------|---------------|------------|---------|---------------------|
| ResNest152           | 99.72%        | 99.52%     | ~0.485  | Best F1 Score       |
| DenseNet161          | High          | High       | ~0.485  | Similar results     |
| InceptionResNetV2    | High          | High       | ~0.485  | Lowest Val Loss     |
| EfficientNet-b3      | High          | Slightly lower | ~0.485 | Low Train Loss  |

---

## 📁 Project Structure
```bash
├── crop_eyes.py                 # ✂️ Crops detected eyes from YOLO predictions
├── eye.yaml                     # 📜 YOLOv11n model configuration
├── eyes.yaml                    # 📜 YOLOv11n dataset configuration
├── pipeline.py                  # 🔄 Main detection → segmentation pipeline
├── predict_segmentation.py      # 🖼 Runs segmentation on cropped eye images
├── segmentation.py              # 🧩 Defines/trains segmentation models (U-Net + encoders)
├── show_mask.py                 # 👁 Visualizes predicted segmentation masks
├── test_segmentation.py         # 🧪 Tests segmentation model performance
├── train_w_pretained.py         # 📈 Trains YOLO model using pretrained weights
├── to_seg.py                    # 🔄 Converts detection outputs into segmentation dataset format
├── aug_for_seg_imgaug.py        # 🎨 Applies imgaug-based augmentations for segmentation data
```
---

## ⚙️ Installation
Install dependencies:
```bash
pip install torch torchvision ultralytics albumentations imgaug opencv-python segmentation-models-pytorch
```
---

## 🚀 YOLOv11n Eye Detection Training   
This project uses YOLOv11n via the Ultralytics Python API for training the eye detection model.
- **📌 Before training, ensure:
  	•	eye.yaml is placed in:
```bash
/Users/<username>/Desktop/y/ultralytics/ultralytics/cfg/models/11/eye.yaml
```
•	eyes.yaml is placed in:
```bash
/Users/<username>/Desktop/y/ultralytics/ultralytics/cfg/datasets/eyes.yaml
```
•	Dataset paths in eyes.yaml point to your train/validation folders.   
###💻 Example Training Script
```bash
from ultralytics import YOLO

# Load YOLOv11n model with custom architecture
model = YOLO("/Users/<username>/Desktop/y/ultralytics/ultralytics/cfg/models/11/eye.yaml")

# Train model using dataset configuration
results = model.train(
    model=model,
    data="/Users/<username>/Desktop/y/ultralytics/ultralytics/cfg/datasets/eyes.yaml",
    epochs=10,
    imgsz=614,
    batch=8,
    device='mps'  # ⚡ Use MPS for Apple Silicon
)
```

---

## 🔄 Workflow
- **1️⃣ Crop Eyes from Images
```bash
python crop_eyes.py --source /path/to/images --weights best.pt --output cropped_eyes/
```
- **2️⃣ Prepare Data for Segmentation
```bash
python to_seg.py --input cropped_eyes/ --output segmentation_dataset/
```
- **3️⃣ Apply Augmentation for Segmentation Dataset
```bash
python aug_for_seg_imgaug.py --input segmentation_dataset/ --output segmentation_aug/
```
- **4️⃣ Train Segmentation Model
```bash
python segmentation.py --encoder resnest152 --epochs 50
```
- **5️⃣ Run the Full Detection → Segmentation Pipeline
```bash
python pipeline.py --image input.jpg --detection_weights best.pt --segmentation_weights best_seg.pth
```
- **6️⃣ Visualize Segmentation Mask
```bash
python show_mask.py --image input.jpg --mask mask.png
```


---

## 📝 Notes
- **📂 Place trained YOLO weights (best.pt) in the detection script directory before running. 
- **📂 Place trained segmentation weights (best_seg.pth) in the segmentation scripts directory before running.
- **⚙️ Adjust imgsz and batch according to your hardware.
- **🌍 For better real-world performance, increase dataset diversity with lighting, angle, and quality variations.

---
