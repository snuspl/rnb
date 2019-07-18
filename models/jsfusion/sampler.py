from nvvl import Sampler
import numpy as np

class FixedSampler(Sampler):
  def __init__(self, num_frames):
    self.num_frames = num_frames

  def _sample(self, length, num_frames):
    if length <= self.num_frames:
      return range(length)
    else:
      return np.linspace(0, length, self.num_frames, endpoint=False, dtype=np.int32)

  def sample(self, length):
    return self._sample(length, self.num_frames)
