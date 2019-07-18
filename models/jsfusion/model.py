from models.jsfusion.module import ResNetFeatureExtractor
from models.jsfusion.module import MCModel
from models.jsfusion.sampler import FixedSampler

from runner_model import RunnerModel
from video_path_provider import VideoPathIterator
from itertools import cycle
from torchvision import transforms
import torch
import nvvl
import os

class JsFusionVideoPathIterator(VideoPathIterator):
  def __init__(self):
    super(JsFusionVideoPathIterator, self).__init__()

    videos = []
    video_dir = os.path.join(os.environ['LSMDC_PATH'], 'mp4s')
    for video in os.listdir(video_dir):
      videos.append(os.path.join(video_dir, video))
  
    if len(videos) <= 0:
      raise Exception('No video available.')

    self.videos_iter = cycle(videos)

  def __iter__(self):
    return self.videos_iter 

class JsFusionLoader(RunnerModel):
  """Impl of loading video frames using NVVL, for the R(2+1)D model."""
  def __init__(self, device):
    self.loader = nvvl.RnBLoader(width=224, height=224,
                                 consecutive_frames=1, device_id=device.index,
                                 sampler=FixedSampler(num_frames=40))

    samples = [
        os.path.join(os.environ['LSMDC_PATH'], 'mp4s/1004_Juno_00.00.32.849-00.00.35.458.mp4'),
        os.path.join(os.environ['LSMDC_PATH'], 'mp4s/1004_Juno_00.00.35.642-00.00.45.231.mp4'),
        os.path.join(os.environ['LSMDC_PATH'], 'mp4s/1004_Juno_00.00.49.801-00.00.59.450.mp4')]

    # warm up GPU with a few inferences
    for sample in samples:
      self.loader.loadfile(sample)
    for frames in self.loader:
      pass
    self.loader.flush()

  def __call__(self, input):
    _, file_path = input
    self.loader.loadfile(file_path)
    for frames in self.loader:
      pass
    self.loader.flush()


    # frames: (40, 3, 1, 224, 224)
    frames = frames.float()
    frames = frames.permute(0, 2, 1, 3, 4)

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    frames_tmp = []
    for frame in frames:
      frame = torch.squeeze(frame)
      frame /= 255
      frame = transform(frame)
      frames_tmp.append(frame)
    frames = torch.stack(frames_tmp)
    # frames: (40, 3, 224, 224)
    
    filename = os.path.basename(file_path)
    out = ((frames,), filename)
    return out

  def __del__(self):
    self.loader.close()

  def input_shape(self):
    return None

  @staticmethod
  def output_shape():
    return ((40, 3, 224, 224),)


class ResNetRunner(RunnerModel):
  def __init__(self, device, num_frames = 40):
    super(ResNetRunner, self).__init__(device)
    self.model = ResNetFeatureExtractor(num_frames).to(device)
    self.model.float()
    self.model.eval()

  def input_shape(self):
    return ((40, 3, 224, 224),)

  @staticmethod
  def output_shape():
    return ((1, 40, 2048),)

  def __call__(self, input):
    return self.model(input)
    

class MCModelRunner(RunnerModel):
  def __init__(self, device, num_frames = 40):
    super(MCModelRunner, self).__init__(device)
    self.model = MCModel(device).to(device)
    self.model.float()
    self.model.eval()

  def input_shape(self):
    return ((1, 40, 2048),)

  def __call__(self, input):
    return self.model(input)

  @staticmethod
  def output_shape():
    return ((1,),)

