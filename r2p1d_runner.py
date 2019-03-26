"""Runner implementation for the R(2+1)D model.
"""
def runner(frame_queue,
           job_id, g_idx, r_idx,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import numpy as np
  import time
  import torch
  from models.r2p1d.network import R2Plus1DClassifier
  from rnb_logging import logname

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


        filename_queue_wait = []
        frame_extraction = []
        frame_queue_wait = []
        inter_gpu_comm = []
        neural_net = []

        with sta_bar_value.get_lock():
          sta_bar_value.value += 1
        if sta_bar_value.value == sta_bar_total:
          sta_bar_semaphore.release()
        sta_bar_semaphore.acquire()
        sta_bar_semaphore.release()

        while True:
          tpl = frame_queue.get()
          if tpl is None:
            break

          time_runner_start = time.time()
          video, time_enqueue_filename, time_loader_start, time_enqueue_data = tpl

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

          # there should be a nicer way to keep all these time measurements...
          filename_queue_wait.append((time_loader_start - time_enqueue_filename) * 1000)
          frame_extraction.append((time_enqueue_data - time_loader_start) * 1000)
          frame_queue_wait.append((time_runner_start - time_enqueue_data) * 1000)
          inter_gpu_comm.append((time_inference_start - time_runner_start) * 1000)
          neural_net.append((time_inference_finish - time_inference_start) * 1000)

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()

  # write statistics AFTER the barrier so that
  # throughput is not affected by unnecessary file I/O
  with open(logname(job_id, g_idx, r_idx), 'w') as f:
    for i in range(len(filename_queue_wait)):
      f.write('%f %f %f %f %f\n' % (filename_queue_wait[i],
                                    frame_extraction[i],
                                    frame_queue_wait[i],
                                    inter_gpu_comm[i],
                                    neural_net[i]))

  # quick summary of the statistics gathered
  print('G%dR%d Average filename queue wait time: %f ms' % \
      (g_idx, r_idx, np.mean(filename_queue_wait[10:])))
  print('G%dR%d Average frame extraction time: %f ms' % \
      (g_idx, r_idx, np.mean(frame_extraction[10:])))
  print('G%dR%d Average frame queue wait time: %f ms' % \
      (g_idx, r_idx, np.mean(frame_queue_wait[10:])))
  print('G%dR%d Average inter-GPU data transmission time: %f ms' % \
      (g_idx, r_idx, np.mean(inter_gpu_comm[10:])))
  print('G%dR%d Average neural net time: %f ms' % \
      (g_idx, r_idx, np.mean(neural_net[10:])))
