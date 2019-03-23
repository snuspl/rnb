from nvvl import Sampler
import random

class R2P1DSampler(Sampler):
  def __init__(self, clip_length=8, num_clips=10):
    self.clip_length = clip_length
    self.num_clips = num_clips

  def _sample(self, length, num_clips):
    if num_clips == 0:
      return None

    if self.clip_length * num_clips > length:
      return self._sample(length, num_clips - 1)

    clip_plus_interval_length = int(float(length) / num_clips)
    interval_length = clip_plus_interval_length - self.clip_length
    leniency = interval_length + (length - clip_plus_interval_length * num_clips)

    start = random.randint(0, leniency)
    frame_indices = [range(start + i * clip_plus_interval_length,
                           start + i * clip_plus_interval_length + 1)
                     for i in range(num_clips)]

    return [item for sublist in frame_indices for item in sublist]

  def sample(self, length):
    return self._sample(length, self.num_clips)
