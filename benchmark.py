"""Main entry point for the video analytics inference benchmark.

This PyTorch benchmark spawns client, loader, and runner processes to perform
video inference in a pipelined fashion. The implementation of each process is
expected to be written in individual modules; this file only provides a bare
backbone of how the overall procedure works. Note that the client, loader,
and runner processes do not necessarily need to be single processes.


          video filenames             video frame tensors
(client) -----------------> (loader) ---------------------> (runner)
               queue                         queue
"""
if __name__ == '__main__':
  # placing these imports before the if statement results in a
  # "context has already been set" RuntimeError
  from torch import multiprocessing as mp
  # https://pytorch.org/docs/stable/notes/multiprocessing.html#sharing-cuda-tensors
  mp.set_start_method('spawn')

  import argparse
  import os
  import sys
  import time
  from datetime import datetime as dt
  from torch.multiprocessing import Queue, Process, Semaphore, Value

  # change these if you want to use different client/loader/runner impls
  from rnb_logging import logmeta
  from control import TerminationFlag
  from client import client
  from r2p1d_loader import loader
  from r2p1d_runner import runner

  parser = argparse.ArgumentParser()
  parser.add_argument('-mi', '--mean_interval_ms',
                      help='Mean event interval time (Poisson), milliseconds',
                      type=int, default=100)
  parser.add_argument('-g', '--gpus', help='Number of GPUs to use',
                      type=int, default=1)
  parser.add_argument('-r', '--replicas_per_gpu',
                      help='Number of replicas per GPU', type=int, default=1)
  parser.add_argument('-b', '--batch_size', help='Video batch size per replica',
                      type=int, default=1)
  parser.add_argument('-v', '--videos', help='Total number of videos to run',
                      type=int, default=2000)
  parser.add_argument('-l', '--loaders', help='Number of loader processes to spawn',
                      type=int, default=1)
  parser.add_argument('-qs', '--queue_size',
                      help='Maximum queue size for inter-process queues',
                      type=int, default=500)
  args = parser.parse_args()
  print('Args:', args)

  job_id = '%s-mi%d-g%d-r%d-b%d-v%d-l%d-qs%d' % (dt.today().strftime('%y%m%d_%H%M%S'),
                                                 args.mean_interval_ms,
                                                 args.gpus,
                                                 args.replicas_per_gpu,
                                                 args.batch_size,
                                                 args.videos,
                                                 args.loaders,
                                                 args.queue_size)

  # assume homogeneous placement of runners
  # in case of a heterogeneous placement, this needs to be changed accordingly
  num_runners = args.gpus * args.replicas_per_gpu

  # barrier to ensure all processes start at the same time
  sta_bar_semaphore = Semaphore(0)
  sta_bar_value = Value('i', 0)
  # runners + loaders + one client + one main process (this one)
  sta_bar_total = num_runners + args.loaders + 2

  # barrier to ensure all processes finish at the same time
  fin_bar_semaphore = Semaphore(0)
  fin_bar_value = Value('i', 0)
  fin_bar_total = num_runners + args.loaders + 2

  # global counter for tracking the total number of videos processed
  # all processes will exit once the counter reaches args.videos
  global_inference_counter = Value('i', 0)

  # global integer flag for checking job termination
  # any process can alter this value to broadcast a termination signal
  termination_flag = Value('i', TerminationFlag.UNSET)

  # queue between client and loader
  filename_queue = Queue(args.queue_size)
  # queue between loader and runner
  frame_queue = Queue(args.queue_size)

  process_client = Process(target=client,
                           args=(filename_queue, args.mean_interval_ms,
                                 args.loaders, termination_flag,
                                 sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                 fin_bar_semaphore, fin_bar_value, fin_bar_total))

  process_loader_list = [Process(target=loader,
                                 args=(filename_queue, frame_queue, num_runners, l,
                                       termination_flag,
                                       sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                       fin_bar_semaphore, fin_bar_value, fin_bar_total))
                         for l in range(args.loaders)]

  process_runner_list = []
  for g in range(args.gpus):
    for r in range(args.replicas_per_gpu):
      process_runner_list.append(Process(target=runner,
                                         args=(frame_queue,
                                               job_id, g, r, global_inference_counter, args.videos,
                                               termination_flag,
                                               sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                               fin_bar_semaphore, fin_bar_value, fin_bar_total)))


  for p in [process_client] + process_loader_list + process_runner_list:
    p.start()

  # we should be able to hide this whole semaphore mess in a
  # single Barrier class or something...
  with sta_bar_value.get_lock():
    sta_bar_value.value += 1
  if sta_bar_value.value == sta_bar_total:
    sta_bar_semaphore.release()
  sta_bar_semaphore.acquire()
  sta_bar_semaphore.release()

  # we can exclude initialization time from the throughput measurement
  # by starting to measure time after the start barrier and not before
  time_start = time.time()
  print('START! %f' % time_start)

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()

  time_end = time.time()
  print('FINISH! %f' % time_end)
  total_time = time_end - time_start
  print('That took %f seconds' % total_time)


  print('Waiting for child processes to return...')
  for p in [process_client] + process_loader_list + process_runner_list:
    p.join()
  

  with open(logmeta(job_id), 'w') as f:
    f.write('Args: %s\n' % str(args))
    f.write('%f %f\n' % (time_start, time_end))
    f.write('Termination flag: %d\n' % termination_flag.value)
