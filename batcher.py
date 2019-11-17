from rnb_logging import TimeCardList
from runner_model import RunnerModel
import torch

class Batcher(RunnerModel):
  def __init__(self, device, batch=1):
    super(Batcher, self).__init__(device)

    self.stacked_tensors = []
    self.stacked_time_cards = []
    self.batch = batch

  @staticmethod
  def output_shape():
    return ((15, 3, 8, 112, 112),)

  def __call__(self, tensors, non_tensors, time_card):
    if self.batch <= 1:
      return tensors, non_tensors, time_card

    tensor = tensors[0]
    self.stacked_tensors.append(tensor)
    self.stacked_time_cards.append(time_card)

    if len(self.stacked_tensors) < self.batch:
      return None, None, None

    tensor_batch = torch.cat(self.stacked_tensors, dim=0)
    time_card_list = TimeCardList(self.stacked_time_cards)

    self.stacked_tensors = []
    self.stacked_time_cards = []

    return (tensor_batch,), None, time_card_list
