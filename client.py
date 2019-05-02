"""Basic client implementation for the video analytics inference benchmark.

Reads video names from a hard-coded file path and sends them to the filename
queue, one at a time. The interval time between enqueues is sampled from an
exponential distribution, to model video inference queries as a Poisson process.
"""
def client(filename_queue, beta, num_loaders, termination_flag,
           sta_bar_semaphore, sta_bar_value, sta_bar_total,
           fin_bar_semaphore, fin_bar_value, fin_bar_total):
  # PyTorch seems to have an issue with sharing modules between
  # multiple processes, so we just do the imports here and
  # not at the top of the file
  import os
  import time
  from numpy.random import exponential
  from queue import Full
  from control import TerminationFlag
  from rnb_logging import TimeCard

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

  with sta_bar_value.get_lock():
    sta_bar_value.value += 1
  if sta_bar_value.value == sta_bar_total:
    sta_bar_semaphore.release()
  sta_bar_semaphore.acquire()
  sta_bar_semaphore.release()

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

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()