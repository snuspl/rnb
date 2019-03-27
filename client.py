"""Basic client implementation for the video analytics inference benchmark.

Reads video names from a hard-coded file path and sends them to the filename
queue, one at a time. The interval time between enqueues is sampled from an
exponential distribution, to model video inference queries as a Poisson process.
"""
def client(filename_queue, beta, num_videos,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import os
  import time
  from numpy.random import exponential

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

  if len(videos) < num_videos:
    print('Available # videos %d < Requested # videos %d' % (len(videos), num_videos))
    print('Will repeat videos to match quota!')
    repeat_count = int(num_videos / len(videos)) + 1
    videos = videos * repeat_count
  videos = videos[:num_videos]

  with sta_bar_value.get_lock():
    sta_bar_value.value += 1
  if sta_bar_value.value == sta_bar_total:
    sta_bar_semaphore.release()
  sta_bar_semaphore.acquire()
  sta_bar_semaphore.release()

  for video in videos:
    # enqueue filename with the current time
    filename_queue.put((video, time.time()))
    time.sleep(exponential(float(beta) / 1000)) # milliseconds --> seconds

  # mark the end of the input stream
  filename_queue.put(None)

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()