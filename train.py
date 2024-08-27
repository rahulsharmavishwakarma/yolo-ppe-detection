from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/teamspace/studios/this_studio/crop_dataset/config.yaml", epochs=100, imgsz=640)