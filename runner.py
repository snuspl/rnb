"""Runner implementation for executing neural networks on the RnB benchmark.
"""
NUM_EXIT_MARKERS = 10
NUM_SUMMARY_SKIPS = 10
def runner(input_queue, output_queue, print_summary,
           job_id, g_idx, r_idx, instance_idx,
           global_inference_counter, num_videos,
           termination_flag, step_idx,
           sta_bar, fin_bar,
           model_module_path, shared_input_tensors, shared_output_tensors,
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


        is_final_step = output_queue is None
        if is_final_step:
          # collect incoming time measurements for later logging
          time_card_summary = TimeCardSummary()

        # keep track of the next position to write output tensors
        shared_output_tensor_counter = 0

        # Create placeholder tensor to copy values from shared input tensors.
        # In case the model does not provide any tensor shape, then we do not
        # make any placeholders.
        shapes = model.input_shape()
        if shapes is not None:
          tensor_input_placeholder = \
              tuple(torch.empty(*shape, dtype=torch.float32).cuda()
                    for shape in shapes)

        sta_bar.wait()

        if print_summary:
          progress_bar = tqdm(total = num_videos)
          old_global_inference_counter_value = 0

        while termination_flag.value == TerminationFlag.UNSET:
          tpl = input_queue.get()
          if tpl is None:
            break

          (signal, non_tensor_inputs), time_card = tpl

          time_card.record('runner%d_start' % step_idx)

          if signal is not None:
            # we need to copy values from the designated shared input tensor
            instance_idx, tensor_idx = signal
            tensor_event = shared_input_tensors[instance_idx][tensor_idx]

            # Under normal circumstances, the event should not be set yet.
            # However, this may not be true if the job is terminating, in which
            # case we immediately exit.
            if tensor_event.event.is_set() and \
                 termination_flag.value != TerminationFlag.UNSET:
               break

            # This is basically a device-to-device memcpy if the source tensors
            # are coming from a different device. If not, then this op becomes
            # a memcpy within the same device.
            for placeholder, shared_tensor in zip(tensor_input_placeholder,
                                                  tensor_event.tensors):
              placeholder.copy_(shared_tensor)

            # release the shared tensor to be reused later
            tensor_event.event.set()

          else:
            # this process does not use the shared tensor mechanism
            tensor_input_placeholder = None

          time_card.record('inference%d_start' % step_idx)

          tensor_outputs, non_tensor_outputs = \
              model((tensor_input_placeholder, non_tensor_inputs))
          stream.synchronize()

          if shared_output_tensors is not None:
            # we need to copy the results into a shared output tensor
            tensor_event = shared_output_tensors[shared_output_tensor_counter]

            # check to see if the tensor has been released or not
            # TODO #59: if this tensor is not ready, then check another one
            tensor_event.event.wait()

            for tensor_output, shared_tensor in zip(tensor_outputs,
                                                    tensor_event.tensors):
              shared_tensor.copy_(tensor_output)

            tensor_event.event.clear()

          time_card.record('inference%d_finish' % step_idx)


          if is_final_step:
            # increment the inference counter
            with global_inference_counter.get_lock():
              global_inference_counter.value += 1

              if print_summary:
                new_counter_value = global_inference_counter.value
                if new_counter_value > old_global_inference_counter_value:
                  progress_bar.update(new_counter_value - old_global_inference_counter_value)
                  old_global_inference_counter_value = new_counter_value

              if global_inference_counter.value == num_videos:
                print('Finished processing %d videos' % num_videos)
                termination_flag.value = TerminationFlag.TARGET_NUM_VIDEOS_REACHED
              elif global_inference_counter.value > num_videos:
                # we've already reached our goal; abort immediately
                break

            time_card_summary.register(time_card)


          else:
            # this is NOT the final step
            # pass on the intermediate tensor to the next step
            try:
              if shared_output_tensors is not None:
                # pass a Signal object for accessing shared tensors
                signal = Signal(instance_idx, shared_output_tensor_counter)
                shared_output_tensor_counter = \
                    (shared_output_tensor_counter + 1) \
                    % len(shared_output_tensors)
              else:
                # no need to pass any signals, just enqueue empty signal
                signal = None
              output_queue.put_nowait(((signal, non_tensor_outputs), time_card))

            except Full:
              print('[WARNING] Queue between runner step %d and %d is full. '
                    'Aborting...' % (step_idx, step_idx+1))
              termination_flag.value = TerminationFlag.FRAME_QUEUE_FULL
              break


        # the termination flag has been raised
        if not is_final_step:
          # mark the end of the input stream
          try:
            for _ in range(NUM_EXIT_MARKERS):
              output_queue.put_nowait(None)
          except Full:
            pass

        if shared_input_tensors is not None:
          # release all shared input tensors in case any process from the
          # previous step is waiting for a tensor to be released
          for instance_tensors in shared_input_tensors:
            for protected_tensor in instance_tensors:
              protected_tensor.event.set()

  fin_bar.wait()
  if output_queue is not None:
    output_queue.cancel_join_thread()

  if is_final_step:
    # write statistics AFTER the barrier so that
    # throughput is not affected by unnecessary file I/O
    with open(logname(job_id, g_idx, r_idx), 'w') as f:
      time_card_summary.save_full_report(f)

    # quick summary of the statistics gathered
    # we skip the first few inferences for stable results
    if print_summary:
      time_card_summary.print_summary(NUM_SUMMARY_SKIPS)
      progress_bar.close()

