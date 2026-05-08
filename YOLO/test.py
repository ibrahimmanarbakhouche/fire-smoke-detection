from ultralytics import YOLO

model = YOLO("C:/Users/borha/Desktop/DL Project V1/archive/YOLO_runs/dfire_model/weights/best.pt")

model.predict(
    source="C:/Users/borha/Desktop/DL Project V1/archive/data/test/images",
    save=True
)
