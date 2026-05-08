from ultralytics import YOLO

model = YOLO("C:/Users/borha/Desktop/DL Project V1/archive/YOLO_runs/dfire_model/weights/best.pt")

metrics = model.val(
    data="C:/Users/borha/Desktop/DL Project V1/archive/data.yaml",
    split="test",
    workers=0
)


print(metrics)