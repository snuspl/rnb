"""Runner implementation for executing neural networks on the RnB benchmark.
"""
NUM_EXIT_MARKERS = 10
NUM_SUMMARY_SKIPS = 10
def runner(input_queue, output_queue,
           job_id, g_idx, r_idx,
           termination_flag, step_idx,
           sta_bar, fin_bar,
           model_module_path, shared_tensors,
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

        shared_tensor_counter = 0
        if step_idx == 1:
          input_placeholder = torch.empty(10, 3, 8, 112, 112, dtype=torch.float32).cuda()
        while termination_flag.value == TerminationFlag.UNSET:
          tpl = input_queue.get()
          if tpl is None:
            break

          tensor, time_card = tpl
          time_card.record('runner%d_start' % step_idx)

          if isinstance(tensor, torch.Tensor) and tensor.device != device:
            tensor = tensor.to(device=device)
          if step_idx == 1:
            protected_tensor = shared_tensors[tensor]
            try:
              assert not protected_tensor.event.is_set()
            except AssertionError as e:
              if termination_flag.value == TerminationFlag.UNSET:
                raise e
              else:
                break
            input_placeholder.copy_(protected_tensor.tensor)
            protected_tensor.event.set()

          time_card.record('inference%d_start' % step_idx)

          if step_idx == 1:
            outputs = model(input_placeholder)
          else:
            outputs = model(tensor)
          if step_idx == 0:
            protected_tensor = shared_tensors[shared_tensor_counter]
            protected_tensor.event.wait()
            protected_tensor.tensor.copy_(outputs)
            protected_tensor.event.clear()
            del outputs
          del tensor
          stream.synchronize()
          time_card.record('inference%d_finish' % step_idx)

          # this is NOT the final step
          # pass on the intermediate tensor to the next step
          try:
            if step_idx == 0:
              output_queue.put_nowait((shared_tensor_counter, time_card))
              shared_tensor_counter = (shared_tensor_counter + 1) % len(shared_tensors)
            else:
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

        for protected_tensor in shared_tensors:
          protected_tensor.event.set()

  fin_bar.wait()
  output_queue.cancel_join_thread()

