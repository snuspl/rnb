import torch
from collections import namedtuple
from torch.multiprocessing import Event
from utils.class_utils import load_class


class TerminationFlag:
  """An enum class for representing various termination states."""
  UNSET = -1
  TARGET_NUM_VIDEOS_REACHED = 0
  FILENAME_QUEUE_FULL = 1
  FRAME_QUEUE_FULL = 2


class SharedQueues:
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


class TensorEvent:
  """Basically a tuple of several torch.Tensors and a multiprocessing.Event.

  The Tensors can be used as "shared tensors" for passing intermediate tensors
  across processes.

  The Event should be used to signal that the consumer process has finished
  reading from the Tensors. When writing values to Tensors, the producer
  process should first check if Tensors are free, by calling event.wait(). If
  the Tensors are indeed free, then event.wait() will return at once. If not,
  then event.wait() will block until the consumer process calls event.set().
  Thus, the consumer should make sure that it calls event.set() AFTER the
  Tensors' contents have been copied to a safe area, such as the consumer's own
  local tensors.
  """
  def __init__(self, shapes, device, dtype=torch.float32):
    self.tensors = tuple(torch.empty(*shape, dtype=dtype, device=device)
                         for shape in shapes)
    self.event = Event()
    self.event.set()


class SharedTensors:
  """Manages intermediate tensors that are passed across steps in the benchmark.

  Args:
    pipeline: The whole pipeline info parsed from the input configuration file
    num_tensors_per_process: The number of shared output tensors that are given
        to each process, for writing tensor values. A big value allows
        processes to produce many tensors before having to block, but requires
        a lot of GPU memory. A small value saves memory, but results in early
        blocking. Note that if a step outputs several tensors during each
        iteration, then this class allocates separate memory for each tensor,
        but still treats them as one tensor when comparing the count with
        num_tensors_per_process.
  """
  def __init__(self, pipeline, num_tensors_per_process):
    # self.tensors is a 3-level list of TensorEvents, e.g.,
    # [
    #   None,                (the first step does not need shared input tensors)
    #   [                                    (shared tensors between step 0 & 1)
    #     [tensorEvent000, tensorEvent001, ...] (outputs of process 0 in step 0)
    #     [tensorEvent010, tensorEvent011, ...] (outputs of process 1 in step 0)
    #     [tensorEvent020, tensorEvent021, ...] (outputs of process 2 in step 0)
    #   ],

    #   [                                    (shared tensors between step 1 & 2)
    #     [tensorEvent100, tensorEvent101, ...] (outputs of process 0 in step 1)
    #     [tensorEvent110, tensorEvent111, ...] (outputs of process 1 in step 1)
    #     [tensorEvent120, tensorEvent121, ...] (outputs of process 2 in step 1)
    #   ],
    #   ...,
    #   [None, None, ...]    (the last step does not need shared output tensors)
    # ]
    self.tensors = [None]

    # we exclude the last step since the last step does not need output tensors
    for i, step in enumerate(pipeline):
      is_final = i == len(pipeline) - 1
      # load the model class to check the output tensor shape of this step
      model_module_path = step['model']
      model_class = load_class(model_module_path)
      shapes = model_class.output_shape()

      if shapes is None:
        # this step does not need shared output tensors
        step_output_tensors = [None for _ in step(['gpus'])]

      else:
        og_shape = shapes[0]
        shapes = ((5, *og_shape[1:]),)
        step_output_tensors = []
        for gpu in step['gpus']:
          device = torch.device('cuda:%d' % gpu)
          tensors = [TensorEvent(shapes, device)
                     for _ in range(num_tensors_per_process if not is_final else 1)]
          step_output_tensors.append(tensors)

      self.tensors.append(step_output_tensors)

  def get_tensors(self, step_idx, instance_idx):
    """Returns the shared input tensors and output tensors for a given process.

    The shared input tensors are returned as a 2-level list, containing the
    output tensors of all processes of the previous step. On the other hand,
    the output tensors are returned as a 1-level list, since this process does
    not need to access the output tensors of other processes from the same step.
    """
    return self.tensors[step_idx], self.tensors[step_idx + 1][instance_idx]
           

# An integer tuple for accessing tensors from SharedTensors.
Signal = namedtuple('Signal', ['instance_idx', 'tensor_idx'])
