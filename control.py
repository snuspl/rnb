import math
import torch
from collections import namedtuple
from torch.multiprocessing import Event, Array
from utils.class_utils import load_class

# default number of shared tensors given to each process for writing outputs
DEFAULT_NUM_SHARED_TENSORS = 10


class TerminationFlag:
  """An enum class for representing various termination states."""
  UNSET = -1
  TARGET_NUM_VIDEOS_REACHED = 0
  FILENAME_QUEUE_FULL = 1
  FRAME_QUEUE_FULL = 2


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

  This class also includes an Array object living on shared memory, consisting
  of integers for indicating the valid region in each tensor. For example, if
  a process uses only 3 rows of a 4-row tensor, then the corresponding entry
  in the Array would be set to 3. Later, when values are read from the tensor
  by another process, that process would first check the Array value and know
  that it can ignore the final row.
  """
  def __init__(self, shapes, device, dtype=torch.float32):
    self.tensors = tuple(torch.empty(*shape, dtype=dtype, device=device)
                         for shape in shapes)
    self.event = Event()
    self.event.set()
    self.valid_batch_sizes = Array('i', len(shapes))


def get_segmented_shapes(shapes, num_segments):
  if shapes is None or num_segments == 1:
    return shapes

  # create smaller shared tensors if segment-based parallel
  # execution is applied
  new_shapes = []
  for shape in shapes:
    batch_size = shape[0]
    if num_segments > batch_size:
      raise ValueError('num_segments %d must be <= tensor batch size %d' %
                       (num_segments, batch_size))

    # We set the shared tensor size to match the largest segment, in
    # case segments have uneven shapes. For example, a 10-row tensor
    # would be divided into 3 segments of 4, 3, and 3 rows each, and
    # the corresponding shared tensor would have 4 rows.
    segment_size = math.ceil(batch_size / num_segments)
    new_shapes.append((segment_size, *shape[1:]))

  return new_shapes


class SharedQueuesAndTensors:
  """Manages intermediate queues & tensors that connect steps in the benchmark.

  Args:
    pipeline: The whole pipeline info parsed from the input configuration file
    queue_class: The class to use when instantiating queues
        (e.g., multiprocessing.Queue, torch.multiprocessing.Queue)
    queue_size: Maximum number of items each queue can hold
  """
  def __init__(self, pipeline, queue_class, queue_size):
    self.filename_queue = queue_class(queue_size)
    self.num_steps = len(pipeline)

    # self.queue_indices is a two-level list of tuples. Basically, it extracts
    # all 'in_queue' and 'out_queues' entries from the pipeline for later use.
    # E.g.,
    # [
    #   [
    #     (step0_group0_in_queue, step0_group0_out_queue_list),
    #     (step0_group1_in_queue, step0_group1_out_queue_list),
    #     ...
    #   ],
    #   [
    #     (step1_group0_in_queue, step1_group0_out_queue_list),
    #     (step1_group1_in_queue, step1_group1_out_queue_list),
    #     ...
    #   ],
    #   ...
    # ]
    self.queue_indices = []

    # self.queues is a list of dictionaries holding all queues.
    # E.g.,
    # [
    #   {0: queue_step0_0, 1: queue_step0_1, ... }, // output queues of step 0
    #   {0: queue_step1_0, 1: queue_step1_1, ... }, // output queues of step 1
    #   {0: queue_step2_0, 1: queue_step2_1, ... }, // output queues of step 2
    #   ...
    # ]
    self.queues = []

    # self.tensors is a four-level list holding all tensors.
    # Level 0: step / Level 1: group / Level 2: instance /
    # Level 3: index (< num_shared_tensors)
    self.tensors = []

    # fill in self.queue_indices, self.queues, and self.tensors for all steps
    for step_idx, step in enumerate(pipeline):
      step_queue_indices = []
      step_queues = {}
      step_tensors = []

      # load the model class to check the output tensor shape of this step
      model_module_path = step['model']
      model_class = load_class(model_module_path)
      shapes = model_class.output_shape()

      # update the shape in case we use segmentation
      num_segments = step.get('num_segments', 1)
      shapes = get_segmented_shapes(shapes, num_segments)

      num_tensors_per_process = step.get('num_shared_tensors',
                                         DEFAULT_NUM_SHARED_TENSORS)

      for group_idx, group in enumerate(step['queue_groups']):
        in_queue_idx = group.get('in_queue', None)
        out_queue_indices = group.get('out_queues', None)

        step_queue_indices.append((in_queue_idx, out_queue_indices))

        # nothing more to do if this is the final step
        if step_idx == len(pipeline) - 1:
          continue

        for out_queue_idx in out_queue_indices:
          # avoid creating duplicate queues for the same queue index
          if out_queue_idx in step_queues: continue
          step_queues[out_queue_idx] = queue_class(queue_size)

        group_tensors = []
        for gpu in group['gpus']:
          device = torch.device('cuda:%d' % gpu)
          tensors = [TensorEvent(shapes, device)
                     for _ in range(num_tensors_per_process)]
          group_tensors.append(tensors)
        step_tensors.append(group_tensors)


      self.queue_indices.append(step_queue_indices)
      self.queues.append(step_queues)
      self.tensors.append(step_tensors)

  def get_filename_queue(self):
    return self.filename_queue

  def get_queues(self, step_idx, group_idx):
    in_queue_idx, out_queue_indices = self.queue_indices[step_idx][group_idx]

    # input queue is always the filename queue for the first step
    in_queue = self.filename_queue if step_idx == 0 \
               else self.queues[step_idx - 1][in_queue_idx]

    # the last step does not need an output queue
    # otherwise, fetch all queues that correspond to the output queue indices
    out_queues = None if step_idx == self.num_steps - 1 \
                 else [self.queues[step_idx][out_queue_idx]
                       for out_queue_idx in out_queue_indices]

    return in_queue, out_queues

  def get_tensors(self, step_idx, group_idx, instance_idx):
    in_queue_idx, out_queue_indices = self.queue_indices[step_idx][group_idx]

    if step_idx == 0:
      # the first step does not require any shared input tensors
      in_tensors = None
    else:
      # check all groups from the previous step that write to my input queue ...
      input_groups = []
      for prev_group_idx, (_, prev_out_queue_indices) in \
          enumerate(self.queue_indices[step_idx - 1]):
        if in_queue_idx in prev_out_queue_indices:
          input_groups.append(prev_group_idx)

      # ... and fetch the tensors from those groups
      in_tensors = {input_group_idx: self.tensors[step_idx - 1][input_group_idx]
                    for input_group_idx in input_groups}


    # the last step does not need shared ouptut tensors
    out_tensors = None if step_idx == self.num_steps - 1 \
                  else self.tensors[step_idx][group_idx][instance_idx]

    return in_tensors, out_tensors


# An integer tuple for accessing tensors from SharedTensors.
Signal = namedtuple('Signal', ['group_idx', 'instance_idx', 'tensor_idx'])
