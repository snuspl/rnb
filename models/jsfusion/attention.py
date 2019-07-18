import torch
import math

MIN_TIMESCALE=1.0
MAX_TIMESCALE=1.0e4

def add_timing_signal_nd(num_frames, video_channels):
  shape = [1, num_frames, video_channels]
  num_dims = len(shape) - 2
  channels = shape[-1]

  position = torch.tensor(range(num_frames), dtype=torch.float32)
  position = torch.unsqueeze(position, dim=1)

  num_timescales = channels // (num_dims * 2)
  log_timescale_increment = math.log(MAX_TIMESCALE / MIN_TIMESCALE) / (num_timescales - 1)
  inv_timescales = []
  for i in range(num_timescales):
    inv_timescales.append(1.0 * math.exp(-float(i) * log_timescale_increment))
  inv_timescales = torch.tensor(inv_timescales, dtype=torch.float32)
  inv_timescales = torch.unsqueeze(inv_timescales, dim=0)

  scaled_time = position.matmul(inv_timescales)
  signal = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
  signal = torch.unsqueeze(signal, 0)

  return signal
