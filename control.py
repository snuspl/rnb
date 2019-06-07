class TerminationFlag:
  """An enum class for representing various termination states."""
  UNSET = -1
  TARGET_NUM_VIDEOS_REACHED = 0
  FILENAME_QUEUE_FULL = 1
  FRAME_QUEUE_FULL = 2


class BenchmarkQueues:
  def __init__(self, queue_class, queue_size, pipeline, per_gpu_queue):
    self.per_gpu_queue = per_gpu_queue
    self.filename_queue = queue_class(queue_size)
    self.num_steps = len(pipeline)

    gpus = set(pipeline[0]['gpus'])
    if per_gpu_queue:
      self.tensor_queues = [{gpu: queue_class(queue_size) for gpu in gpus}
                            for step_idx in range(self.num_steps - 1)]
    else:
      self.tensor_queues = [{0:queue_class(queue_size)}
                            for step_idx in range(self.num_steps - 1)]


  def get_tensor_queue(self, step_idx, gpu_idx):
    queue_idx = gpu_idx if self.per_gpu_queue else 0
    prev_queue = self.filename_queue if step_idx == 0 \
                 else self.tensor_queues[step_idx - 1][queue_idx]
    next_queue = None if step_idx == self.num_steps - 1 \
                 else self.tensor_queues[step_idx][queue_idx]

    return prev_queue, next_queue

  def get_filename_queue(self):
    return self.filename_queue
