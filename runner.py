"""Runner implementation for executing neural networks on the RnB benchmark.
"""
NUM_EXIT_MARKERS = 10
NUM_SUMMARY_SKIPS = 10
def runner(input_queue, output_queues, selector_path, print_summary,
           job_id, g_idx, group_idx, instance_idx,
           global_inference_counter, num_videos,
           termination_flag, step_idx,
           sta_bar, fin_bar,
           model_module_path, num_segments,
           shared_input_tensors, shared_output_tensors,
           **model_kwargs):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import numpy as np
  import torch
  from queue import Empty, Full
  from tqdm import tqdm
  from rnb_logging import logname, TimeCardSummary, TimeCard
  from control import TerminationFlag, Signal
  from utils.class_utils import load_class

  # We need to explicitly set the default device of this process to be g_idx.
  # Otherwise, this process will request memory on GPU 0 for a short time right
  # before it terminates (for some bizarre reason), which might lead to an OOM
  # on GPU 0 if many runners are present.
  if g_idx >= 0:
    torch.cuda.set_device(g_idx)

  # Use our own CUDA stream to avoid synchronizing with other processes
  # This is a no-op if g_idx is negative
  with torch.cuda.device(g_idx):
    # use CPU if g_idx is negative
    device = torch.device('cuda:%d' % g_idx) if g_idx >= 0 \
             else torch.device('cpu')
    # do not create a CUDA stream if we use CPU
    stream = torch.cuda.Stream(device=g_idx) if g_idx >= 0 else None

    # this is a no-op if stream is None
    with torch.cuda.stream(stream):
      with torch.no_grad():
        # load model instance using the given module path
        model_class = load_class(model_module_path)
        model = model_class(device, **model_kwargs)


        is_final_step = output_queues is None
        if is_final_step:
          # collect incoming time measurements for later logging
          time_card_summary = TimeCardSummary()
        else:
          # instantitate selector for choosing which queue to write outputs to
          selector_class = load_class(selector_path)
          selector = selector_class(len(output_queues))

        # keep track of the next position to write output tensors
        shared_output_tensor_counter = 0

        # Create placeholder tensor to copy values from shared input tensors.
        # In case there are no shared input tensors, we do not
        # make any placeholders either.
        if shared_input_tensors is not None:
          # pick any random tensor from input tensors, since
          # all tensors have the same shape anyway
          tensor_event = list(shared_input_tensors.values())[0][0][0]
          tensor_input_placeholder = \
              tuple(torch.zeros_like(tensor, device=device)
                    for tensor in tensor_event.tensors)

        sta_bar.wait()

        if print_summary:
          progress_bar = tqdm(total = num_videos)
          old_global_inference_counter_value = 0

        while termination_flag.value == TerminationFlag.UNSET:
          tpl = input_queue.get()
          if tpl is None:
            break

          signal, non_tensor_inputs, time_card = tpl

          time_card.add_gpu(g_idx)
          time_card.record('runner%d_start' % step_idx)

          if signal is not None:
            # we need to copy values from the designated shared input tensor
            signal_group_idx, signal_instance_idx, tensor_idx = signal
            tensor_event = shared_input_tensors[signal_group_idx][signal_instance_idx][tensor_idx]

            # Under normal circumstances, the event should not be set yet.
            # However, this may not be true if the job is terminating, in which
            # case we immediately exit.
            if tensor_event.event.is_set() and \
                 termination_flag.value != TerminationFlag.UNSET:
               break


            tensor_inputs = []
            # This is basically a device-to-device memcpy if the source tensors
            # are coming from a different device. If not, then this op becomes
            # a memcpy within the same device.
            for idx, (placeholder, shared_tensor) in \
                enumerate(zip(tensor_input_placeholder, tensor_event.tensors)):
              valid_batch_size = tensor_event.valid_batch_sizes[idx]

              # only copy the valid regions of the shared input tensor
              placeholder[:valid_batch_size].copy_(
                  shared_tensor[:valid_batch_size])
              tensor_inputs.append(placeholder[:valid_batch_size])

            # release the shared tensor to be reused later
            tensor_event.event.set()

          else:
            # this process does not use the shared tensor mechanism
            tensor_inputs = None

          time_card.record('inference%d_start' % step_idx)

          tensor_outputs, non_tensor_outputs, time_card = \
              model(tensor_inputs, non_tensor_inputs, time_card)
          if stream is not None:
            stream.synchronize()

          # We assume that None for time_card means this runner has no outputs.
          # In this case, nothing is passed to the next step, and the global
          # inference counter is not incremented.
          if time_card is None:
            continue

          time_card.record('inference%d_finish' % step_idx)

          if shared_output_tensors is not None:
            # partition the output tensor into segments
            for segment_idx in range(num_segments):
              tensor_segment_outputs = []

              for tensor_output in tensor_outputs:
                # The indexing method below divides the batch by the number of
                # segments (the division remainders are placed evenly across
                # segments, starting from the first one).
                # For example, a batch of 11 is partitioned into [0:4] (4 rows),
                # [4:8] (4 rows), and [8:11] (3 rows).
                q, r = divmod(tensor_output.shape[0], num_segments)
                batch_start_idx = q * segment_idx + min(segment_idx, r)
                batch_end_idx = q * (segment_idx+1) + min(segment_idx+1, r)

                tensor_segment_outputs.append(
                    tensor_output[batch_start_idx:batch_end_idx])

              # we need to copy the results into a shared output tensor
              counter = (shared_output_tensor_counter + segment_idx) % \
                  len(shared_output_tensors)
              tensor_event = shared_output_tensors[counter]

              # check to see if the tensor has been released or not
              # TODO #59: if this tensor is not ready, then check another one
              tensor_event.event.wait()

              # similar to when we copied values from shared input tensors,
              # we make sure to copy only the valid regions of the output tensor
              for idx, (tensor_segment_output, shared_tensor) in \
                  enumerate(zip(tensor_segment_outputs, tensor_event.tensors)):
                valid_batch_size = tensor_segment_output.shape[0]
                shared_tensor[:valid_batch_size].copy_(tensor_segment_output)
                tensor_event.valid_batch_sizes[idx] = valid_batch_size

              tensor_event.event.clear()


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
            output_queue_idx = selector.select(tensor_outputs, non_tensor_outputs, time_card)
            output_queue = output_queues[output_queue_idx]
            try:
              for segment_idx in range(num_segments):
                # we create a child of the current TimeCard if segment-based
                # parallel execution has been applied
                forked_tc = time_card.fork(segment_idx) if num_segments > 1 \
                            else time_card

                if shared_output_tensors is not None:
                  # pass a Signal object for accessing shared tensors
                  signal = Signal(group_idx, instance_idx, shared_output_tensor_counter)
                  shared_output_tensor_counter = \
                      (shared_output_tensor_counter + 1) \
                      % len(shared_output_tensors)
                else:
                  # no need to pass any signals, just enqueue empty signal
                  signal = None
                output_queue.put_nowait((signal, non_tensor_outputs, forked_tc))

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
              for output_queue in output_queues:
                output_queue.put_nowait(None)
          except Full:
            pass

        if shared_input_tensors is not None:
          # release all shared input tensors in case any process from the
          # previous step is waiting for a tensor to be released
          for group_tensors in shared_input_tensors.values():
            for instance_tensors in group_tensors:
              for protected_tensor in instance_tensors:
                protected_tensor.event.set()

  fin_bar.wait()
  if output_queues is not None:
    for output_queue in output_queues:
      output_queue.cancel_join_thread()

  if is_final_step:
    # write statistics AFTER the barrier so that
    # throughput is not affected by unnecessary file I/O
    with open(logname(job_id, g_idx, group_idx, instance_idx), 'w') as f:
      time_card_summary.save_full_report(f)

    # quick summary of the statistics gathered
    # we skip the first few inferences for stable results
    if print_summary:
      time_card_summary.print_summary(NUM_SUMMARY_SKIPS)
      progress_bar.close()

