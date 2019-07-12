import torch
from torch.multiprocessing import Event
from utils.class_utils import load_class


class TerminationFlag:
  """An enum class for representing various termination states."""
  UNSET = -1
  TARGET_NUM_VIDEOS_REACHED = 0
  FILENAME_QUEUE_FULL = 1
  FRAME_QUEUE_FULL = 2


class BenchmarkQueues:
  """Manages intermediate queues that connect steps in the benchmark.

  Args:
    queue_class: The class to use when instantiating queues
        (e.g., multiprocessing.Queue, torch.multiprocessing.Queue)
    queue_size: Maximum number of items each queue can hold
    pipeline: The whole pipeline info parsed from the input configuration file
    per_gpu_queue: True if GPU-local queues should be used,
        or False if global queues should be used
  """
  def __init__(self, queue_class, queue_size, pipeline, per_gpu_queue):
    self.per_gpu_queue = per_gpu_queue
    self.filename_queue = queue_class(queue_size)
    self.aggregator_queue = queue_class(queue_size)
    self.num_steps = len(pipeline)

    # self.tensor_queues is a list of dictionaries, e.g.,
    # [
    #   {0: filename_queue, 1: filename_queue}, (queue between client and step0)
    #   {0: q_step01_gpu0, 1: q_step01_gpu1}, (set of queues between step 0 & 1)
    #   {0: q_step12_gpu0, 1: q_step12_gpu1}, (set of queues between step 1 & 2)
    #   ...
    #   {0: None, 1: None} (the last step does not need an output queue)
    # ]
    # in case we use global queues, each dictionary will hold only one queue:
    # [
    #   {0: filename_queue}, (queue between client and step 0)
    #   {0: q_step01}, (global queue between step 0 & 1)
    #   {0: q_step12}, (global queue between step 1 & 2)
    #   {0: None} (the last step does not need an output queue)
    # ]
    if per_gpu_queue:
      # we assume that all steps use the same set of gpus
      gpus = set(pipeline[0]['gpus'])
      self.tensor_queues = [{gpu: queue_class(queue_size) for gpu in gpus}
                            for step_idx in range(self.num_steps - 1)]

      # The first step will receive filenames from client via filename_queue.
      # Unlike tensor queues, filename_queue is always a global queue so we
      # insert the same entry (self.filename_queue) for all gpu indices.
      self.tensor_queues.insert(0, {gpu: self.filename_queue for gpu in gpus})

      # The last step does need an output queue, so we pass None.
      self.tensor_queues.append({gpu: self.aggregator_queue for gpu in gpus})

    else:
      # There is no need for differentiating queues according to gpu index,
      # since there is only one global queue (per step) anyway.
      # Thus, instead of using gpu index as the dictionary key, we simply set
      # the number 0 as the only key.
      # Note that we could even just abandon the dictionary type and do
      # something like [queue_class(queue_size) for _ in range(...)],
      # but we keep the dictionary type to simplify the logic of later
      # choosing prev_queue and next_queue in get_tensor_queue().
      self.tensor_queues = [{0: queue_class(queue_size)}
                            for step_idx in range(self.num_steps - 1)]

      # The first step will receive filenames from client via filename_queue.
      self.tensor_queues.insert(0, {0: self.filename_queue})

      # The last step does need an output queue, so we pass None.
      self.tensor_queues.append({0: self.aggregator_queue})

  def get_tensor_queue(self, step_idx, gpu_idx):
    queue_idx = gpu_idx if self.per_gpu_queue else 0
    prev_queue = self.tensor_queues[step_idx][queue_idx]
    next_queue = self.tensor_queues[step_idx + 1][queue_idx]

    return prev_queue, next_queue

  def get_filename_queue(self):
    return self.filename_queue

  def get_aggregator_queue(self):
    return self.aggregator_queue


class ProtectedTensor:
  def __init__(self, shape, device, dtype=torch.float32):
    self.tensor = torch.empty(*shape, dtype=dtype, device=device)
    self.event = Event()
    self.event.set()

  def __str__(self):
    return str(self.tensor)


class BenchmarkTensors:
  def __init__(self, pipeline, num_tensors_per_process):
    self.tensors = []

    for step in pipeline:
      step_tensors = []

      model_module_path = step['model']
      model_class = load_class(model_module_path)
      shape = model_class.output_shape()
      for gpu in step['gpus']:
        device = torch.device('cuda:%d' % gpu)
        tensors = [ProtectedTensor(shape, device)
                   for _ in range(num_tensors_per_process)]
        step_tensors.append(tensors)

      self.tensors.append(step_tensors)
