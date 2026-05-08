from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="C:/Users/user/Desktop/DL Project V1/archive/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=10,
    device="cuda",
    workers=0,
    amp=True,
    project="C:/Users/user/Desktop/DL Project V1/archive/YOLO_runs",
    name="dfire_model"
)