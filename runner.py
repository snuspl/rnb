"""Runner implementation for executing neural networks on the RnB benchmark.
"""
def runner(input_queue, output_queue, num_exit_markers, print_summary,
           job_id, g_idx, r_idx, global_inference_counter, num_videos,
           termination_flag, step_idx,
           start_idx, end_idx, 
           sta_bar, fin_bar,
           model_module_path):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import numpy as np
  import torch
  from queue import Empty, Full
  from rnb_logging import logname, TimeCardSummary
  from control import TerminationFlag

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
        delimiter_idx = model_module_path.rfind('.')
        module_path = model_module_path[:delimiter_idx]
        model_name = 'R2P1DLayerRunner' 
        #model_name = model_module_path[delimiter_idx+1:] 
        module = __import__(module_path, fromlist=(model_name))
        model_class = getattr(module, model_name)
        print("=======RUNNER 39=======", model_class)
        model = model_class(device, start_idx, end_idx)
        input_shape = model.input_shape()
        print("========RUNNER 42=========", type(model), input_shape)
        # first "warm up" the model with a few sample inferences
        tmp = torch.randn(*input_shape, dtype=torch.float32).cuda()
        for _ in range(3):
          _ = model(tmp)
          stream.synchronize()
        del tmp


        if output_queue is None:
          # this is the final step
          # collect incoming time measurements for later logging
          time_card_summary = TimeCardSummary()

        sta_bar.wait()

        while termination_flag.value == TerminationFlag.UNSET:
          tpl = input_queue.get()
          if tpl is None:
            break

          tensor, time_card = tpl
          time_card.record('runner%d_start' % step_idx)

          if tensor.device != device:
            tensor = tensor.to(device=device)
          time_card.record('inference%d_start' % step_idx)

          outputs = model(tensor)
          stream.synchronize()
          time_card.record('inference%d_finish' % step_idx)


          if output_queue is None:
            # this is the final step
            # increment the inference counter
            with global_inference_counter.get_lock():
              global_inference_counter.value += 1

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
              output_queue.put_nowait((outputs, time_card))
            except Full:
              print('[WARNING] Queue between runner step %d and %d is full. '
                    'Aborting...' % (step_idx, step_idx+1))
              termination_flag.value = TerminationFlag.FRAME_QUEUE_FULL
              break


        # the termination flag has been raised
        if output_queue is not None:
          # this is NOT the final step
          # mark the end of the input stream
          try:
            for _ in range(num_exit_markers):
              output_queue.put_nowait(None)
          except Full:
            pass

  fin_bar.wait()

  if output_queue is None:
    # this is the final step

    # write statistics AFTER the barrier so that
    # throughput is not affected by unnecessary file I/O
    with open(logname(job_id, g_idx, r_idx), 'w') as f:
      time_card_summary.save_full_report(f)

    # quick summary of the statistics gathered
    # we skip the first few inferences for stable results
    NUM_SKIPS = 10
    if print_summary:
      time_card_summary.print_summary(NUM_SKIPS)

  # We've observed cases where the loader processes do not exit until
  # all tensors spawned from the loaders are removed from scope (even if they
  # reach the end of the `loader` function).
  # We clear the input queue here so loaders can exit successfully.
  # Note that it doesn't matter if this cleanup takes long, because the
  # throughput measurement has already been done at the finish barrier above.
  try:
    while True:
      input_queue.get_nowait()
  except Empty:
    pass
