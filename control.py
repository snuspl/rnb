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
    self.num_steps = len(pipeline)

    # self.tensor_queues is a list of dictionaries, e.g.,
    # [
    #   {0: q_step01_gpu0, 1: q_step01_gpu1}, (set of queues between step 0 & 1)
    #   {0: q_step12_gpu0, 1: q_step12_gpu1}, (set of queues between step 1 & 2)
    # ]
    # in case we use global queues, each dictionary will hold only one queue:
    # [
    #   {0: q_step01}, (global queue between step 0 & 1)
    #   {0: q_step12}, (global queue between step 1 & 2)
    # ]
    if per_gpu_queue:
      # we assume that all steps use the same set of gpus
      gpus = set(pipeline[0]['gpus'])
      self.tensor_queues = [{gpu: queue_class(queue_size) for gpu in gpus}
                            for step_idx in range(self.num_steps - 1)]
    else:
      self.tensor_queues = [{0:queue_class(queue_size)}
                            for step_idx in range(self.num_steps - 1)]

  def get_tensor_queue(self, step_idx, gpu_idx):
    queue_idx = gpu_idx if self.per_gpu_queue else 0

    # the first step will receive filenames from the client via filename_queue
    prev_queue = self.filename_queue if step_idx == 0 \
                 else self.tensor_queues[step_idx - 1][queue_idx]

    # the last step does not need an output queue, so we pass None
    next_queue = None if step_idx == self.num_steps - 1 \
                 else self.tensor_queues[step_idx][queue_idx]

    return prev_queue, next_queue

  def get_filename_queue(self):
    return self.filename_queue
