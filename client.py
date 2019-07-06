"""Client implementations for the video analytics inference benchmark.

The Client simulates creating inference requests, assuming that
videos are available on the server. We can evaluate different workload
characteristics by implementing different clients; for example, we have poisson_client
that generates requests for a single video at a time with random intervals, and
bulk_client that generates requests for all videos at once.
"""
NUM_EXIT_MARKERS = 10

def poisson_client(video_path_iterator, filename_queue, beta, termination_flag,
                   sta_bar, fin_bar):
  """Sends loaded video to the filename queue, one at a time.

  The interval time between enqueues is sampled from an exponential distribution, 
  to model video inference queries as a Poisson process.
  """
  import time
  from numpy.random import exponential
  from queue import Full
  from control import TerminationFlag
  from rnb_logging import TimeCard
  from utils.class_utils import load_class

  sta_bar.wait()

  for video_path in load_class(video_path_iterator).__init__():
    if termination_flag.value != TerminationFlag.UNSET:
      break

    # create TimeCard instance to measure the time of key events
    time_card = TimeCard()
    time_card.record('enqueue_filename')

    try:
      filename_queue.put_nowait((video_path, time_card))
    except Full:
      print('[WARNING] Filename queue is full. Aborting...')
      termination_flag.value = TerminationFlag.FILENAME_QUEUE_FULL
      break
    time.sleep(exponential(float(beta) / 1000)) # milliseconds --> seconds

  # mark the end of the input stream
  # the loaders should exit by themselves, but we enqueue these markers just in
  # case some loader is waiting on the queue
  try:
    for _ in range(NUM_EXIT_MARKERS):
      filename_queue.put_nowait(None)
  except Full:
    # if the queue is full, then we don't have to do anything because
    # the loaders will not be blocked at queue.get() and eventually exit
    # on their own
    pass

  fin_bar.wait()
  filename_queue.cancel_join_thread()

def bulk_client(video_path_iterator_class, filename_queue, num_videos, termination_flag,
                sta_bar, fin_bar):
  """Sends videos to the filename queue in bulk, as many as specified by the argument num_videos.

  This implementation is mainly for measuring maximum throughput where latency is not a primary metric. 
  """
  import time
  from queue import Full
  from control import TerminationFlag
  from rnb_logging import TimeCard
  from utils.class_utils import load_class

  sta_bar.wait()

  video_count = 0
  for video_path in load_class(video_path_iterator).__init__():
    if video_count >= num_videos:
      break

    video_count += 1

    # create TimeCard instance to measure the time of key events
    time_card = TimeCard()
    time_card.record('enqueue_filename')

    try:
      filename_queue.put_nowait((video_path, time_card))
    except Full:
      print('[WARNING] Filename queue is full. Aborting...')
      termination_flag.value = TerminationFlag.FILENAME_QUEUE_FULL
      break

  # mark the end of the input stream
  # the loaders should exit by themselves, but we enqueue these markers just in
  # case some loader is waiting on the queue
  try:
    for _ in range(NUM_EXIT_MARKERS):
      filename_queue.put_nowait(None)
  except Full:
    # if the queue is full, then we don't have to do anything because
    # the loaders will not be blocked at queue.get() and eventually exit
    # on their own
    pass

  fin_bar.wait()
  filename_queue.cancel_join_thread()
