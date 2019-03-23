"""Loader implementation for loading videos to the R(2+1)D model.

For each video, we sample a certain number of clips (default: 10), which in turn
are consisted of a certain number of consecutive frames (default: 8).
http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf

The frames are downsampled (default: 112x112) and sent to the data queue, as
tensors of shape (num_clips, 3, consecutive_frames, width, height).
"""
def loader(filename_queue, data_queue,
           num_runners,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import time
  import torch
  import nvvl
  from r2p1d_sampler import R2P1DSampler

  # the GPU id does NOT necessarily have to be 0
  # Use our own CUDA stream to avoid synchronizing with other processes
  with torch.cuda.device(0):
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
      with torch.no_grad():
        loader = nvvl.RnBLoader(width=112, height=112,
                                consecutive_frames=8, device_id=0,
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

        while True:
          tpl = filename_queue.get()
          if tpl is None:
            break

          tloader_start = time.time()
          filename, tenqueue_filename = tpl

          loader.loadfile(filename)
          for frames in loader:
            pass
          loader.flush()

          # enqueue frames with past and current timestamps
          data_queue.put((frames,
                          tenqueue_filename,
                          tloader_start,
                          time.time()))

        # mark the end of the input stream
        for _ in range(num_runners):
          data_queue.put(None)

        loader.close()

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()
