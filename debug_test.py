import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

weights = 'yolov5s.pt'
device = '0' if torch.cuda.is_available() else 'cpu'
device = select_device(device)

model = attempt_load(weights, map_location=device)
model.to(device).eval()
model.half()

names = model.module.names if hasattr(
            model, 'module') else model.names

print(model)