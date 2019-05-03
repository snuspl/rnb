"""Simple test for demonstrating KernelTracker from the CUPTI bridge."""
import torch

from torch import nn
from utils import cupti

device = torch.device('cuda:0')
tensor = torch.randn(10, 2000, dtype=torch.float32, device=device)

kernel_tracker = cupti.KernelTracker()
layer = nn.Linear(2000, 100, bias=True).to(device)

result = layer(tensor)

for kernel in kernel_tracker.get_kernel_names():
  print(kernel)
kernel_tracker.reset()
