"""Client implementations for the video analytics inference benchmark.
   The Client simulates creating inference requests, assuming that 
   the video is available in the server. We can evaluate different workload
   characteristics by implementing different client; for example, we have poisson_client
   that generates requests for single video at a time with random interval, and 
   bulk_client that generates requests for all videos at once.
"""
def load_videos():
  """Helper function that reads video names from a hard-coded file path.
  """
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import os

  # file directory is assumed to be like:
  # root/
  #   label1/
  #     video1
  #     video2
  #     ...
  #   label2/
  #     video3
  #     video4
  #     ...
  #   ...
  root = '/cmsdata/ssd0/cmslab/Kinetics-400/sparta'
  videos = []
  for label in os.listdir(root):
    for video in os.listdir(os.path.join(root, label)):
      videos.append(os.path.join(root, label, video))

  if len(videos) <= 0:
    raise Exception('No video available.')
  return videos

def poisson_client(filename_queue, beta, num_loaders, termination_flag,
        sta_bar, fin_bar):
  """
  Sends loaded video to the filename queue, one at a time. 
  The interval time between enqueues is sampled from an exponential distribution, 
  to model video inference queries as a Poisson process.
  """
  import time
  from numpy.random import exponential
  from queue import Full
  from control import TerminationFlag
  from rnb_logging import TimeCard

  videos = load_videos()

  sta_bar.wait()

  video_idx = 0
  while termination_flag.value == TerminationFlag.UNSET:
    video = videos[video_idx]
    # come back to the front of the list if we're at the end
    video_idx = (video_idx + 1) % len(videos)

    # create TimeCard instance to measure the time of key events
    time_card = TimeCard()
    time_card.record('enqueue_filename')

    try:
      filename_queue.put_nowait((video, time_card))
    except Full:
      print('[WARNING] Filename queue is full. Aborting...')
      termination_flag.value = TerminationFlag.FILENAME_QUEUE_FULL
      break
    time.sleep(exponential(float(beta) / 1000)) # milliseconds --> seconds

  # mark the end of the input stream
  # the loaders should exit by themselves, but we enqueue these markers just in
  # case some loader is waiting on the queue
  try:
    for _ in range(num_loaders):
      filename_queue.put_nowait(None)
  except Full:
    # if the queue is full, then we don't have to do anything because
    # the loaders will not be blocked at queue.get() and eventually exit
    # on their own
    pass

  fin_bar.wait()

def bulk_client(filename_queue, beta, num_loaders, num_videos, termination_flag,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total):
  """
  Sends videos to the filename queue in bulk, as many as specified by the argument num_videos. 
  This implementation is mainly for measuring maximum throughput where latency is not a primary metric. 
  """
  import time
  from queue import Full
  from control import TerminationFlag
  from rnb_logging import TimeCard

  assert beta == 0
  videos = load_videos()

  sta_bar.wait()

  video_idx = 0
  while video_idx < num_videos:
    video = videos[video_idx]
    # come back to the front of the list if we're at the end
    video_idx = (video_idx + 1) % len(videos)

    # create TimeCard instance to measure the time of key events
    time_card = TimeCard()
    time_card.record('enqueue_filename')

    try:
      filename_queue.put_nowait((video, time_card))
    except Full:
      print('[WARNING] Filename queue is full. Aborting...')
      termination_flag.value = TerminationFlag.FILENAME_QUEUE_FULL
      break

  # mark the end of the input stream
  # the loaders should exit by themselves, but we enqueue these markers just in
  # case some loader is waiting on the queue
  try:
    for _ in range(num_loaders):
      filename_queue.put_nowait(None)
  except Full:
    # if the queue is full, then we don't have to do anything because
    # the loaders will not be blocked at queue.get() and eventually exit
    # on their own
    pass

  fin_bar.wait()
