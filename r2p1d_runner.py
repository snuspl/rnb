"""Runner implementation for the R(2+1)D model.
"""
def runner(frame_queue,
           job_id, g_idx, r_idx, global_inference_counter, num_videos,
           termination_flag,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import numpy as np
  import time
  import torch
  from models.r2p1d.network import R2Plus1DClassifier
  from queue import Empty
  from rnb_logging import logname, Termination

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

        model = R2Plus1DClassifier(num_classes=400,
                                   layer_sizes=[2,2,2,2]).to(device)
        ckpt = torch.load('/cmsdata/ssd0/cmslab/Kinetics-400/ckpt/model_data.pth.tar',
                          map_location=device)
        model.load_state_dict(ckpt['state_dict'])

        # first "warm up" the model with a few sample inferences
        tmp = torch.randn(10, 3, 8, 112, 112, dtype=torch.float32).cuda()
        for _ in range(3):
          _ = model(tmp)
          stream.synchronize()
        del tmp


        time_enqueue_filename_list = []
        time_loader_start_list = []
        time_enqueue_frame_list = []
        time_runner_start_list = []
        time_inference_start_list = []
        time_inference_finish_list = []

        with sta_bar_value.get_lock():
          sta_bar_value.value += 1
        if sta_bar_value.value == sta_bar_total:
          sta_bar_semaphore.release()
        sta_bar_semaphore.acquire()
        sta_bar_semaphore.release()

        while termination_flag.value == 0:
          tpl = frame_queue.get()
          if tpl is None:
            break

          time_runner_start = time.time()
          video, time_enqueue_filename, time_loader_start, time_enqueue_frames = tpl

          if video.device != device:
            video = video.to(device=device)
          time_inference_start = time.time()

          video = video.float()
          # (num_clips, 3, consecutive_frames, width, height)
          # --> (num_clips, consecutive_frames, 3, width, height)
          video = video.permute(0, 2, 1, 3, 4)

          outputs = model(video)
          stream.synchronize()
          time_inference_finish = time.time()

          with global_inference_counter.get_lock():
            global_inference_counter.value += 1

            if global_inference_counter.value == num_videos:
              print('Finished processing %d videos' % num_videos)
              termination_flag.value = Termination.TARGET_NUM_VIDEOS_REACHED
            elif global_inference_counter.value > num_videos:
              # we've already reached our goal; abort immediately
              break

          # there should be a nicer way to keep all these time measurements...
          time_enqueue_filename_list.append(time_enqueue_filename)
          time_loader_start_list.append(time_loader_start)
          time_enqueue_frame_list.append(time_enqueue_frames)
          time_runner_start_list.append(time_runner_start)
          time_inference_start_list.append(time_inference_start)
          time_inference_finish_list.append(time_inference_finish)


  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()

  # write statistics AFTER the barrier so that
  # throughput is not affected by unnecessary file I/O
  with open(logname(job_id, g_idx, r_idx), 'w') as f:
    f.write(' '.join(['enqueue_filename', 'loader_start', 'enqueue_frames',
                      'runner_start', 'inference_start', 'inference_finish']))
    f.write('\n')
    for tpl in zip(time_enqueue_filename_list, time_loader_start_list,
                   time_enqueue_frame_list, time_runner_start_list,
                   time_inference_start_list, time_inference_finish_list):
      f.write(' '.join(map(str, tpl)))
      f.write('\n')

  # quick summary of the statistics gathered
  # we skip the first few inferences for stable results
  NUM_SKIPS = 10
  if g_idx == 0 and r_idx == 0:
    print('Average filename queue wait time: %f ms' % \
        (np.mean((np.array(time_loader_start_list) - np.array(time_enqueue_filename_list))[NUM_SKIPS:]) * 1000))
    print('Average frame extraction time: %f ms' % \
        (np.mean((np.array(time_enqueue_frame_list) - np.array(time_loader_start_list))[NUM_SKIPS:]) * 1000))
    print('Average frame queue wait time: %f ms' % \
        (np.mean((np.array(time_runner_start_list) - np.array(time_enqueue_frame_list))[NUM_SKIPS:]) * 1000))
    print('Average inter-GPU frames transmission time: %f ms' % \
        (np.mean((np.array(time_inference_start_list) - np.array(time_runner_start_list))[NUM_SKIPS:]) * 1000))
    print('Average neural net time: %f ms' % \
        (np.mean((np.array(time_inference_finish_list) - np.array(time_inference_start_list))[NUM_SKIPS:]) * 1000))

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