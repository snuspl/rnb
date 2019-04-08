"""Loader implementation for loading videos to the R(2+1)D model.

For each video, we sample a certain number of clips (default: 10), which in turn
are consisted of a certain number of consecutive frames (default: 8).
http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf

The frames are downsampled (default: 112x112) and sent to the frame queue, as
tensors of shape (num_clips, 3, consecutive_frames, width, height).
"""
def loader(filename_queue, frame_queue,
           num_runners, idx, global_inference_counter, num_videos,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import time
  import torch
  import nvvl
  from r2p1d_sampler import R2P1DSampler

  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(idx):
    device = torch.device('cuda:%d' % idx)
    stream = torch.cuda.Stream(device=idx)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        loader = nvvl.RnBLoader(width=112, height=112,
                                consecutive_frames=8, device_id=idx,
                                sampler=R2P1DSampler(clip_length=8))

        # first "warm up" the loader with a few videos
        samples = [
            '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/0gR5FP7HpZ4_000024_000034.mp4',
            '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/2WowmnRTyqY_000203_000213.mp4',
            '/cmsdata/ssd0/cmslab/Kinetics-400/sparta/laughing/5GXEjJjgGcc_000058_000068.mp4',
        ]

        for sample in samples:
          loader.loadfile(sample)
        for frames in loader:
          pass
        loader.flush()

        with sta_bar_value.get_lock():
          sta_bar_value.value += 1
        if sta_bar_value.value == sta_bar_total:
          sta_bar_semaphore.release()
        sta_bar_semaphore.acquire()
        sta_bar_semaphore.release()

        # Exit the main loop when the counter reaches `num_videos`.
        while global_inference_counter.value < num_videos:
          tpl = filename_queue.get()
          if tpl is None:
            # apparently, the client has already aborted which means the
            # counter has reached `num_videos`; so we exit
            break

          time_loader_start = time.time()
          filename, time_enqueue_filename = tpl

          loader.loadfile(filename)
          # we only load one file, so loader.__iter__ returns only one item
          for frames in loader:
            # in case we load multiple files in the future, then we would
            # actually need to do something within this for loop
            pass
          # close the file since we're done with it
          loader.flush()

          # enqueue frames with past and current timestamps
          frame_queue.put((frames,
                           time_enqueue_filename,
                           time_loader_start,
                           time.time()))

        # mark the end of the input stream
        # the runners should exit by themselves, but we enqueue these markers
        # just in case some runner is waiting on the queue
        if idx == 0:
          for _ in range(num_runners):
            frame_queue.put(None)

        loader.close()

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()
