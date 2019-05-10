"""Simple test for retrieving kernel timestamps using the CUPTI bridge."""
import torch

from torch.nn import Conv2d
from utils import cupti

device = torch.device('cuda:0')
# first layer of
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
model = Conv2d(3, 64, kernel_size=11, stride=4, padding=2).to(device)

# batch size 4
tensor = torch.randn(4, 3, 224, 224, dtype=torch.float32, device=device)

cupti.initialize()
results = model(tensor)
torch.cuda.synchronize()
cupti.flush()

for name, time_start, time_end in cupti.report():
  print(name, time_start, time_end)
