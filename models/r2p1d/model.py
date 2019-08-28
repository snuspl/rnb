import nvvl
import torch
import sys 
import os
from itertools import cycle

from models.r2p1d.sampler import R2P1DSampler
from models.r2p1d.network import R2Plus1DClassifier, SpatioTemporalResBlock
from models.r2p1d.network import R2Plus1DLayerWrapper
from rnb_logging import TimeCard
from runner_model import RunnerModel
from video_path_provider import VideoPathIterator

CKPT_PATH = '/cmsdata/ssd0/cmslab/Kinetics-400/ckpt/model_data.pth.tar'

class R2P1DRunner(RunnerModel):
  """RunnerModel that can create any connected subset of the R(2+1)D model for RnB benchmark runners.
  
  start_index and end_index are assumed to be 1-indexed. 
  """
  # holds the expected tensor input shape for all 5 layers 
  # the first layer expects an input shape of 
  # (10 clips, 3 channels, 8 consecutive frames, 112 pixels for width, 112 pixels for height)
  # input shape for the later layers change like the following as tensors propagate each layer
  input_dict = { 1: (10, 3, 8, 112, 112), 
                 2: (10, 64, 8, 56, 56),
                 3: (10, 64, 8, 56, 56),
                 4: (10, 128, 4, 28, 28),
                 5: (10, 256, 2, 14, 14) }

  def __init__(self, device, start_index=1, end_index=5, num_classes=400, layer_sizes=None, block_type=SpatioTemporalResBlock):
    super(R2P1DRunner, self).__init__(device)
    
    if start_index < 1:
      print('[ERROR] Wrong layer index for the starting layer! The start_index (%d) should be more than or equal to 1.' % start_index) 
      sys.exit()
    
    elif end_index > 5:  
      print('[ERROR] Wrong layer index for the ending layer! The end_index (%d) should be less than or equal to 5.' % end_index)
      sys.exit()
    
    if layer_sizes is None:
      layer_sizes = [2 for _ in range(start_index, end_index+1)]
    self.start_index = start_index
    self.model = R2Plus1DLayerWrapper(start_index, end_index, num_classes, layer_sizes, block_type).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)

    state_dict = {}
    # filter out weights that are not used in this model  
    for i in range(start_index, end_index+1):
      layer = 'res2plus1d.conv%d' % i
 
      state_dict.update({k:v for k, v in ckpt['state_dict'].items() if
                         k.startswith(layer)})
    
    if end_index == 5:
      state_dict.update({k:v for k, v in ckpt['state_dict'].items() if
                         k.startswith('linear')})
    self.model.load_state_dict(state_dict) 

    # warm up GPU with a few inferences
    inp_shape = self.input_shape()
    stream = torch.cuda.current_stream()
    tmp = torch.randn(*inp_shape, dtype=torch.float32).cuda()
    for _ in range(3):
      _ = self.model(tmp)
      stream.synchronize()
    
  def input_shape(self):
    return (self.input_dict[self.start_index],)

  @staticmethod
  def output_shape():
    # TODO #69: the output shape may not be (10, 400),
    # depending on self.end_index; need to change return value accordingly
    return ((10, 400),)

  def __call__(self, tensors, non_tensors, time_card):
    tensor = tensors[0]
    return (self.model(tensor),), None, time_card

class R2P1DVideoPathIterator(VideoPathIterator):
  def __init__(self):
    super(R2P1DVideoPathIterator, self).__init__()

    videos = []
    # file directory is assumed to be like:
    # root/
    #   label1/
    #     video1
    #     video2
    #     ...
    #   label2/
    #     video3
    #     video4
    #     ...
    #   ...
    root = '/cmsdata/ssd0/cmslab/Kinetics-400/sparta'
    for label in os.listdir(root):
      for video in os.listdir(os.path.join(root, label)):
        videos.append(os.path.join(root, label, video))

    if len(videos) <= 0:
      raise Exception('No video available.')

    self.videos_iter = cycle(videos)

  def __iter__(self):
    return self.videos_iter


class R2P1DLoader(RunnerModel):
  """Impl of loading video frames using NVVL, for the R(2+1)D model."""
  def __init__(self, device):
    self.loader = nvvl.RnBLoader(width=112, height=112,
                                 consecutive_frames=8, device_id=device.index,
                                 sampler=R2P1DSampler(clip_length=8))

    samples = [
        '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4',
        '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/2WowmnRTyqY_000203_000213.mp4',
        '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/5GXEjJjgGcc_000058_000068.mp4',
    ]

    # warm up GPU with a few inferences
    for sample in samples:
      self.loader.loadfile(sample)
    for frames in self.loader:
      pass
    self.loader.flush()

  def __call__(self, tensors, non_tensors, time_card):
    filename = non_tensors
    self.loader.loadfile(filename)
    for frames in self.loader:
      pass
    self.loader.flush()

    frames = frames.float()
    frames = frames.permute(0, 2, 1, 3, 4)
    return (frames,), None, time_card

  def input_shape(self):
    return None

  @staticmethod
  def output_shape():
    return ((10, 3, 8, 112, 112),)


class R2P1DSingleStep(RunnerModel):
  """RunnerModel impl that contains all inference logic regarding R(2+1)D.

  This RunnerModel can basically be used to run the R(2+1)D model without any
  pipelining between steps. In terms of code, this class simply merges
  R2P1DLoader with R2P1DLayerRunner, excluding the functionality of specifying
  layer start and end indices (`start_index` and `end_index`).
  """
  input_tensor_shape = (10, 3, 8, 112, 112)

  def __init__(self, device, num_classes=400, layer_sizes=[2,2,2,2],
               block_type=SpatioTemporalResBlock):
    super(R2P1DSingleStep, self).__init__(device)

    # instantiate the main neural network
    self.model = R2Plus1DClassifier(num_classes, layer_sizes, block_type) \
                     .to(device)

    # initalize the model with pre-trained weights
    ckpt = torch.load(CKPT_PATH, map_location=device)
    self.model.load_state_dict(ckpt['state_dict'])


    # prepare the loader for converting videos into frames
    self.loader = nvvl.RnBLoader(width=112, height=112,
                                 consecutive_frames=8, device_id=device.index,
                                 sampler=R2P1DSampler(clip_length=8))


    # warm up the neural network
    stream = torch.cuda.current_stream()
    tmp = torch.randn(*self.input_tensor_shape, dtype=torch.float32).cuda()
    for _ in range(3):
      _ = self.model(tmp)
      stream.synchronize()


    # warm up the loader
    samples = [
        '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4',
        '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/2WowmnRTyqY_000203_000213.mp4',
        '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/5GXEjJjgGcc_000058_000068.mp4',
    ]

    for sample in samples:
      self.loader.loadfile(sample)
    for frames in self.loader:
      pass
    self.loader.flush()
    stream.synchronize()

  def __call__(self, tensors, non_tensors, time_card):
    filename = non_tensors
    self.loader.loadfile(filename)
    for frames in self.loader:
      pass
    self.loader.flush()

    frames = frames.float()
    frames = frames.permute(0, 2, 1, 3, 4)

    return (self.model(frames),), None, time_card

  def input_shape(self):
    return None

  @staticmethod
  def output_shape():
    return ((10, 400),)


class R2P1DAggregator(RunnerModel):
  """RunnerModel that aggregates inference segments produced by R2P1DRunner."""
  def __init__(self, device, aggregate=1):
    super(R2P1DAggregator, self).__init__(device)
    
    # This parameter indicates the expected number of segments per inference
    # instance. This should be equal to num_segments of a previous step.
    self.aggregate = aggregate
    self.results = {}

  def __call__(self, tensors, non_tensors, time_card):
    tensor = tensors[0]
    # We don't really need to store the whole tensor as-is, because the final
    # operation is an average followed by an argmax. We can reduce memory space
    # by simply summing it by the batch dimension, which can be summed again
    # with later incoming segments.
    result = tensor.cpu().numpy().sum(axis=0)

    if self.aggregate == 1:
      # no need to perform any kind of segment aggregation,
      # so just return immediately
      return None, result.argmax(), time_card

    if time_card.id not in self.results:
      self.results[time_card.id] = (result, [time_card])
      # this inference needs to wait for other segments
      return None, None, None

    else:
      prev_result, prev_time_cards = self.results[time_card.id]
      result = prev_result + result
      time_cards = prev_time_cards + [time_card]

      if len(time_cards) < self.aggregate:
        # still waiting for other segments
        self.results[time_card.id] = (result, time_cards)
        return None, None, None
      else:
        # all segments have arrived
        merged_time_card = TimeCard.merge(time_cards)
        return None, result.argmax(), merged_time_card

  def input_shape(self):
    return ((10, 400),)

  @staticmethod
  def output_shape():
    return None
