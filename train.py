from ultralytics import YOLO

model = YOLO("path/to/eye.yaml")
results = model.train(model=model, data="path/to/eyes.yaml", epochs=10, imgsz=614, batch=8, device='mps')
