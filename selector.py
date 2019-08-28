class Selector:
  """Interface class for selecting output queue based on runner outputs."""
  def __init__(self, num_queues):
    pass

  def select(self, tensors, non_tensors, time_card):
    raise NotImplementedError


class RoundRobinSelector(Selector):
  """A simple impl of Selector that chooses queues in a round-robin fashion."""
  def __init__(self, num_queues):
    self.num_queues = num_queues
    self.curr = 0

  def select(self, tensors, non_tensors, time_card):
    self.curr = (self.curr + 1) % self.num_queues
    return self.curr
