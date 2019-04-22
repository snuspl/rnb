import torch

from models.r2p1d.network import R2Plus1DClassifier, SpatioTemporalResBlock
from runner_model import RunnerModel

CKPT_PATH = '/cmsdata/ssd0/cmslab/Kinetics-400/ckpt/model_data.pth.tar'

class R2P1D(RunnerModel):
  """RunnerModel impl of the R(2+1)D model for RnB benchmark runners."""
  def __init__(self, device, num_classes=400, layer_sizes=[2,2,2,2],
                     block_type=SpatioTemporalResBlock):
    super(R2P1D, self).__init__(device)

    self.model = R2Plus1DClassifier(num_classes, layer_sizes, block_type) \
                     .to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    self.model.load_state_dict(ckpt['state_dict'])


  def input_shape(self):
    return (10, 3, 8, 112, 112)  


  def __call__(self, input):
    return self.model(input)
