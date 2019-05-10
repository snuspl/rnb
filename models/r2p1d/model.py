import torch
import sys 

from models.r2p1d.network import R2Plus1DClassifier, SpatioTemporalResBlock
from models.r2p1d.network import R2Plus1DLayerWrapper
from models.r2p1d.network import R2Plus1DLayer12Wrapper
from models.r2p1d.network import R2Plus1DLayer345Wrapper
from runner_model import RunnerModel

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
  input_dict = { 1: (10, 3, 8, 112, 112), 
                 2: (10, 64, 8, 56, 56),
                 3: (10, 64, 8, 56, 56),
                 4: (10, 128, 4, 28, 28),
                 5: (10, 256, 2, 14, 14) }

  def __init__(self, device, start_index=1, end_index=5, num_classes=400, block_type=SpatioTemporalResBlock):
    super(R2P1DLayerRunner, self).__init__(device)
    
    if start_index < 1:
      print('[ERROR] Wrong layer index for the starting layer! The start_index (%d) should be more than or equal to 0.' % start_index) 
      sys.exit()
    
    elif end_index > 5:  
      print('[ERROR] Wrong layer index for the ending layer! The end_index (%d) should be more than or equal to 0.' % end_index)
      sys.exit()
    
    else:
      layer_sizes = [2 for x in range(start_index, end_index+1)]
      self.start_index = start_index
      self.model = R2Plus1DLayerWrapper(start_index, end_index, num_classes, layer_sizes, block_type).to(device)
      ckpt = torch.load(CKPT_PATH, map_location=device)

      state_dict = {}
      # filter out weights that are not used in this model  
      for i in range(start_index, end_index+1):
        layer = 'res2plus1d.conv{}'.format(i)
   
        state_dict.update({k:v for k, v in ckpt['state_dict'].items() if
                          k.startswith(layer)})
      
      state_dict.update({k:v for k, v in ckpt['state_dict'].items() if
                          k.startswith('linear') and end_index == 5})
      self.model.load_state_dict(state_dict) 
    
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
