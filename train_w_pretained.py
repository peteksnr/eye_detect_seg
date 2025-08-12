from ultralytics import YOLO


model = YOLO("/Users/peteksener/Desktop/y/runs/detect/train9/weights/best.pt")
results = model.train(model=model, data="/Users/peteksener/Desktop/y/ultralytics/ultralytics/cfg/datasets/eyes.yaml", epochs=20, imgsz=614, batch=8, device='mps')