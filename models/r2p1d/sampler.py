"""An NVVL Sampler implementation that samples frames for the R(2+1)D model.

Given a video, this sampler returns multiple frame indices that represent the
starting frames of target clips (sub-videos). The positions of clips are
determined according to the original R(2+1)D paper.
http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf

Clips must be spread across the whole video as wide as possible, but at the
same time no two adjacent clips can be closer to each other than any other
adjacent clips. Even with such restrictions, there can generally be several
possible solutions, at which point we just randomly pick one.

Example: sample 3 7-frame clips from a 25-frame video

   01-02-03-04-05-06-07-08-09-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25
S1 |------------------|    |------------------|    |------------------|
S2    |------------------|    |------------------|    |------------------|
S3       |------------------|    |------------------|    |------------------|
"""

from nvvl import Sampler
import random

class R2P1DSampler(Sampler):
  def __init__(self, clip_length=8, num_clips=10):
    """
    Args:
      clip_length: number of frames per clip
      num_clips: number of clips to sample per single video
    """
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
