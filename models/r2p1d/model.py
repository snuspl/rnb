import torch
import sys 

from models.r2p1d.network import R2Plus1DClassifier, SpatioTemporalResBlock
from models.r2p1d.network import R2Plus1DLayerWrapper
from models.r2p1d.network import R2Plus1DLayer12Wrapper
from models.r2p1d.network import R2Plus1DLayer345Wrapper
from runner_model import RunnerModel

import nvvl
from r2p1d_sampler import R2P1DSampler

CKPT_PATH = '/cmsdata/ssd0/cmslab/Kinetics-400/ckpt/model_data.pth.tar'

class R2P1DRunner(RunnerModel):
  """RunnerModel impl of the R(2+1)D model for RnB benchmark runners."""
  def __init__(self, device, num_classes=400, layer_sizes=[2,2,2,2],
                     block_type=SpatioTemporalResBlock):
    super(R2P1DRunner, self).__init__(device)

    self.model = R2Plus1DClassifier(num_classes, layer_sizes, block_type) \
                     .to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    self.model.load_state_dict(ckpt['state_dict'])


  def input_shape(self):
    return (10, 3, 8, 112, 112)


  def __call__(self, input):
    return self.model(input)

class R2P1DLayerRunner(RunnerModel):
  """RunnerModel that can create any connected subset of the R(2+1)D model for RnB benchmark runners.
  
  start_index and end_index are assumed to be 1-indexed. 
  """
  # holds the expected tensor input shape for all 5 layers 
  # the first layer expects an input shape of 
  # (10 clips, 3 channels, 8 consecutive frames, 112 pixels for width, 112 pixels for height)
  # input shape for the later layers change like the following as tensors propagate each layer
  input_dict = { 1: (5, 3, 8, 112, 112), 
                 2: (5, 64, 8, 56, 56),
                 3: (5, 64, 8, 56, 56),
                 4: (5, 128, 4, 28, 28),
                 5: (5, 256, 2, 14, 14) }

  def __init__(self, device, start_index=1, end_index=5, num_classes=400, layer_sizes=None, block_type=SpatioTemporalResBlock):
    super(R2P1DLayerRunner, self).__init__(device)
    
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


    inp_shape = self.input_shape()
    stream = torch.cuda.current_stream()
    tmp = torch.randn(*inp_shape, dtype=torch.float32).cuda()
    for _ in range(3):
      _ = self.model(tmp)
      stream.synchronize()
    
  def input_shape(self):
    return self.input_dict[self.start_index]

  def __call__(self, input):
    return self.model(input)

class R2P1DLayer12Runner(RunnerModel):
  """RunnerModel impl for the first two layers of the R(2+1)D model."""
  def __init__(self, device, layer_size=2, block_type=SpatioTemporalResBlock):
    super(R2P1DLayer12Runner, self).__init__(device)

    self.model = R2Plus1DLayer12Wrapper(layer_size, block_type).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)

    # filter out weights that are not used in this model
    state_dict = {k:v for k,v in ckpt['state_dict'].items() if
                  k.startswith('res2plus1d.conv1') or
                  k.startswith('res2plus1d.conv2')}
    self.model.load_state_dict(state_dict)


  def input_shape(self):
    return (10, 3, 8, 112, 112)


  def __call__(self, input):
    return self.model(input)


class R2P1DLayer345Runner(RunnerModel):
  """RunnerModel impl for the last three layers of the R(2+1)D model."""
  def __init__(self, device, num_classes=400, layer_sizes=[2,2,2],
                     block_type=SpatioTemporalResBlock):
    super(R2P1DLayer345Runner, self).__init__(device)

    self.model = R2Plus1DLayer345Wrapper(num_classes, layer_sizes, block_type) \
                     .to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)

    # filter out weights that are not used in this model
    state_dict = {k:v for k,v in ckpt['state_dict'].items() if
                      k.startswith('res2plus1d.conv3') or
                      k.startswith('res2plus1d.conv4') or
                      k.startswith('res2plus1d.conv5') or
                      k.startswith('linear')}
    self.model.load_state_dict(state_dict)


  def input_shape(self):
    return (10, 64, 8, 56, 56)


  def __call__(self, input):
    return self.model(input)


class R2P1DLoader(RunnerModel):
  def __init__(self, device):
    self.loader = nvvl.RnBLoader(width=112, height=112,
                                 consecutive_frames=8, device_id=device.index,
                                 sampler=R2P1DSampler(clip_length=8))

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


  def __call__(self, input):
    self.loader.loadfile(input)
    for frames in self.loader:
      pass
    self.loader.flush()

    frames = frames.float()
    frames = frames.permute(0, 2, 1, 3, 4)
    return frames


class R2P1D(RunnerModel):
  input_dict = { 1: (10, 3, 8, 112, 112),
                 2: (10, 64, 8, 56, 56),
                 3: (10, 64, 8, 56, 56),
                 4: (10, 128, 4, 28, 28),
                 5: (10, 256, 2, 14, 14) }

  def __init__(self, device, start_index=1, end_index=5, num_classes=400,
               layer_sizes=None, block_type=SpatioTemporalResBlock):
    super(R2P1D, self).__init__(device)

    #############################################
    ####### PREPARE NEURAL NETWORK RUNNER #######
    #############################################
    if start_index < 1:
      print('[ERROR] Wrong layer index for the starting layer! '
            'The start_index (%d) should be more than or equal to 1.'
            % start_index)
      sys.exit()

    elif end_index > 5:
      print('[ERROR] Wrong layer index for the ending layer! '
            'The end_index (%d) should be less than or equal to 5.'
            % end_index)
      sys.exit()

    if layer_sizes is None:
      layer_sizes = [2 for _ in range(start_index, end_index+1)]
    self.start_index = start_index
    self.model = R2Plus1DLayerWrapper(start_index, end_index, num_classes,
                                      layer_sizes, block_type).to(device)
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


    #############################################
    ####### PREPARE NEURAL NETWORK LOADER #######
    #############################################
    self.loader = nvvl.RnBLoader(width=112, height=112,
                                 consecutive_frames=8, device_id=device.index,
                                 sampler=R2P1DSampler(clip_length=8))


    #############################################
    ####### WARM UP NEURAL NETWORK RUNNER #######
    #############################################
    inp_shape = self.input_dict[self.start_index]
    stream = torch.cuda.current_stream()
    tmp = torch.randn(*inp_shape, dtype=torch.float32).cuda()
    for _ in range(3):
      _ = self.model(tmp)
      stream.synchronize()


    #############################################
    ####### WARM UP NEURAL NETWORK LOADER #######
    #############################################
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


  def __call__(self, input):
    self.loader.loadfile(input)
    for frames in self.loader:
      pass
    self.loader.flush()

    frames = frames.float()
    frames = frames.permute(0, 2, 1, 3, 4)

    return self.model(frames)
