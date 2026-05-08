from ultralytics import YOLO
from thop import profile
import torch

model = YOLO("best.pt").model

dummy_input = torch.randn(1, 3, 640, 640)

flops, params = profile(model, inputs=(dummy_input, ))

print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M")