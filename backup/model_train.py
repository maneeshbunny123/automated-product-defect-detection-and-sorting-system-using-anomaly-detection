from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pretrained model
results = model.train(data="E:/My_Project/dataset/data.yaml", epochs=100, imgsz=640)

