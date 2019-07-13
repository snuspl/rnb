"""Runner implementation for executing neural networks on the RnB benchmark.
"""
NUM_EXIT_MARKERS = 10
NUM_SUMMARY_SKIPS = 10
def runner(input_queue, output_queue,
           job_id, g_idx, instance_idx,
           termination_flag, step_idx,
           sta_bar, fin_bar,
           model_module_path, shared_output_tensors, shared_input_tensors,
           **model_kwargs):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import numpy as np
  import torch
  from queue import Empty, Full
  from tqdm import tqdm
  from rnb_logging import logname, TimeCardSummary
  from control import TerminationFlag, Signal
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

        output_shared_tensor_counter = 0
        shape = model.input_shape()
        if step_idx != 0 and shape is not None:
          input_placeholder = torch.empty(*shape, dtype=torch.float32).cuda()

        while termination_flag.value == TerminationFlag.UNSET:
          tpl = input_queue.get()
          if tpl is None:
            break

          signal_or_input, time_card = tpl
          time_card.record('runner%d_start' % step_idx)

          if isinstance(signal_or_input, Signal):
            signal = signal_or_input
            protected_tensor = \
                shared_input_tensors[signal.instance_idx][signal.tensor_idx]

            try:
              assert not protected_tensor.event.is_set()
            except AssertionError as e:
              if termination_flag.value == TerminationFlag.UNSET:
                raise e
              else:
                break

            input_placeholder.copy_(protected_tensor.tensor)
            protected_tensor.event.set()

          else:
            input_placeholder = signal_or_input


          time_card.record('inference%d_start' % step_idx)
          outputs = model(input_placeholder)
          stream.synchronize()

          if shared_output_tensors is not None:
            protected_tensor = shared_output_tensors[output_shared_tensor_counter]
            protected_tensor.event.wait()
            protected_tensor.tensor.copy_(outputs)
            protected_tensor.event.clear()
          time_card.record('inference%d_finish' % step_idx)

          # this is NOT the final step
          # pass on the intermediate tensor to the next step
          try:
            signal = Signal(instance_idx, output_shared_tensor_counter)
            output_queue.put_nowait((signal, time_card))
            output_shared_tensor_counter = (output_shared_tensor_counter + 1) % len(shared_output_tensors)
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

        if shared_input_tensors is not None:
          for instance_tensors in shared_input_tensors:
            for protected_tensor in instance_tensors:
              protected_tensor.event.set()

  fin_bar.wait()
  output_queue.cancel_join_thread()

