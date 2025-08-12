from ultralytics import YOLO

model = YOLO("/Users/peteksener/Desktop/y/ultralytics/ultralytics/cfg/models/11/eye.yaml")
results = model.train(model=model, data="/Users/peteksener/Desktop/y/ultralytics/ultralytics/cfg/datasets/eyes.yaml", epochs=10, imgsz=614, batch=8, device='mps')