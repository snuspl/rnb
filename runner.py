"""Runner implementation for executing neural networks on the RnB benchmark.
"""
NUM_EXIT_MARKERS = 10
NUM_SUMMARY_SKIPS = 10
def runner(input_queue, output_queue,
           job_id, g_idx, r_idx,
           termination_flag, step_idx,
           sta_bar, fin_bar,
           model_module_path,
           **model_kwargs):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import numpy as np
  import torch
  from queue import Empty, Full
  from tqdm import tqdm
  from rnb_logging import logname, TimeCardSummary
  from control import TerminationFlag
  from utils.class_utils import load_class

  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(g_idx):
    device = torch.device('cuda:%d' % g_idx)
    stream = torch.cuda.Stream(device=g_idx)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        # PyTorch seems to have a strange issue of taking a long time to exit
        # the Python process in case it doesn't use the first GPU...
        # so we allocate a small tensor at the first GPU before doing anything
        # to avoid the problem altogether.
        # This issue may have been fixed in the latest PyTorch release.
        # TODO #2: Update PyTorch version
        insurance = torch.randn(1, device=torch.device('cuda:0'))

        # load model instance using the given module path
        model_class = load_class(model_module_path)
        model = model_class(device, **model_kwargs)

        sta_bar.wait()

        while termination_flag.value == TerminationFlag.UNSET:
          tpl = input_queue.get()
          if tpl is None:
            break

          tensor, time_card = tpl
          time_card.record('runner%d_start' % step_idx)

          if isinstance(tensor, torch.Tensor) and tensor.device != device:
            tensor = tensor.to(device=device)
          time_card.record('inference%d_start' % step_idx)

          outputs = model(tensor)
          del tensor
          stream.synchronize()
          time_card.record('inference%d_finish' % step_idx)

          # this is NOT the final step
          # pass on the intermediate tensor to the next step
          try:
            output_queue.put_nowait((outputs, time_card))
          except Full:
            print('[WARNING] Queue between runner step %d and %d is full. '
                  'Aborting...' % (step_idx, step_idx+1))
            termination_flag.value = TerminationFlag.FRAME_QUEUE_FULL
            break


        # the termination flag has been raised
        # mark the end of the input stream
        try:
          for _ in range(NUM_EXIT_MARKERS):
            output_queue.put_nowait(None)
        except Full:
          pass

  fin_bar.wait()
  output_queue.cancel_join_thread()

