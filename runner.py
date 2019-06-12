"""Runner implementation for executing neural networks on the RnB benchmark.
"""
def runner(input_queue, output_queue, print_summary,
           job_id, g_idx, r_idx, global_inference_counter, num_videos,
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
  from rnb_logging import logname, TimeCardSummary, TimeCard
  from control import TerminationFlag

  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(g_idx):
    device = torch.device('cuda:%d' % g_idx)
    stream = torch.cuda.Stream(device=g_idx)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        # load model instance using the given module path
        delimiter_idx = model_module_path.rfind('.')
        module_path = model_module_path[:delimiter_idx]
        model_name = model_module_path[delimiter_idx+1:]
        module = __import__(module_path, fromlist=(model_name))
        model_class = getattr(module, model_name)

        model = model_class(device, **model_kwargs)

        with torch.cuda.device(0):
          insurance = torch.randn(1).cuda()


        sta_bar.wait()

        # if print_summary:
        #   progress_bar = tqdm(total = num_videos)
        #   old_global_inference_counter_value = 0

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

          # if step_idx == 1:
            # outputs = 0
            # time_card = TimeCard(time_card.id)
          stream.synchronize()
          time_card.record('inference%d_finish' % step_idx)

          if step_idx == 0:
            outputs1 = outputs[:5, :, :, :, :]
            outputs2 = outputs[5:, :, :, :, :]
            try:
              tc = TimeCard(time_card.id)
              tc.timings = time_card.timings
              time_card.sub_id = 0
              tc.sub_id = 1
              output_queue.put_nowait((outputs1, time_card))
              output_queue.put_nowait((outputs2, tc))
            except Full:
              print('[WARNING] Queue between runner step %d and %d is full. '
                    'Aborting...' % (step_idx, step_idx+1))
              termination_flag.value = TerminationFlag.FRAME_QUEUE_FULL
              break
            continue

          try:
            output_queue.put_nowait((outputs, time_card))
          except Full:
            print('[WARNING] Queue between runner step %d and %d is full. '
                  'Aborting...' % (step_idx, step_idx+1))
            termination_flag.value = TerminationFlag.FRAME_QUEUE_FULL
            break


        # # the termination flag has been raised
        # if not is_final_step:
        # mark the end of the input stream
        try:
          NUM_EXIT_MARKERS = 10
          for _ in range(NUM_EXIT_MARKERS):
            output_queue.put_nowait(None)
        except Full:
          pass

  fin_bar.wait()
  # if output_queue is not None:
  output_queue.cancel_join_thread()

  # if is_final_step:
  #   # write statistics AFTER the barrier so that
  #   # throughput is not affected by unnecessary file I/O
  #   with open(logname(job_id, g_idx, r_idx), 'w') as f:
  #     time_card_summary.save_full_report(f)

  #   # quick summary of the statistics gathered
  #   # we skip the first few inferences for stable results
  #   NUM_SKIPS = 10
  #   if print_summary:
  #     time_card_summary.print_summary(NUM_SKIPS)
  #     progress_bar.close()
