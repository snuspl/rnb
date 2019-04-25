"""Runner implementation for executing neural networks on the RnB benchmark.
"""
def runner(frame_queue,
           job_id, g_idx, r_idx, global_inference_counter, num_videos,
           termination_flag,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total,
           model_module_path):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import numpy as np
  import torch
  from queue import Empty
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
        model_name = model_module_path[delimiter_idx+1:]
        module = __import__(module_path, fromlist=(model_name))
        model_class = getattr(module, model_name)

        model = model_class(device)
        input_shape = model.input_shape()

        # first "warm up" the model with a few sample inferences
        tmp = torch.randn(*input_shape, dtype=torch.float32).cuda()
        for _ in range(3):
          _ = model(tmp)
          stream.synchronize()
        del tmp


        # collect incoming time measurements for later logging
        time_card_summary = TimeCardSummary()

        with sta_bar_value.get_lock():
          sta_bar_value.value += 1
        if sta_bar_value.value == sta_bar_total:
          sta_bar_semaphore.release()
        sta_bar_semaphore.acquire()
        sta_bar_semaphore.release()

        while termination_flag.value == TerminationFlag.UNSET:
          tpl = frame_queue.get()
          if tpl is None:
            break

          video, time_card = tpl
          time_card.record('runner_start')

          if video.device != device:
            video = video.to(device=device)
          time_card.record('inference_start')

          video = video.float()
          # (num_clips, 3, consecutive_frames, width, height)
          # --> (num_clips, consecutive_frames, 3, width, height)
          video = video.permute(0, 2, 1, 3, 4)

          outputs = model(video)
          stream.synchronize()
          time_card.record('inference_finish')

          with global_inference_counter.get_lock():
            global_inference_counter.value += 1

            if global_inference_counter.value == num_videos:
              print('Finished processing %d videos' % num_videos)
              termination_flag.value = TerminationFlag.TARGET_NUM_VIDEOS_REACHED
            elif global_inference_counter.value > num_videos:
              # we've already reached our goal; abort immediately
              break

          time_card_summary.register(time_card)

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()

  # write statistics AFTER the barrier so that
  # throughput is not affected by unnecessary file I/O
  with open(logname(job_id, g_idx, r_idx), 'w') as f:
    time_card_summary.save_full_report(f)

  # quick summary of the statistics gathered
  # we skip the first few inferences for stable results
  NUM_SKIPS = 10
  if g_idx == 0 and r_idx == 0:
    time_card_summary.print_summary(NUM_SKIPS)

    # We've observed cases where the loader processes do not exit until
    # all tensors spawned from the loaders are removed from scope (even if they
    # reach the end of the `loader` function).
    # We clear the frame queue here so loaders can exit successfully.
    # Note that it doesn't matter if this cleanup takes long, because the
    # throughput measurement has already been done at the finish barrier above.
    try:
      while True:
        frame_queue.get_nowait()
    except Empty:
      pass
