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
  from torch.multiprocessing import SimpleQueue, Process, Semaphore, Value

  # change these if you want to use different client/loader/runner impls
  from rnb_logging import logname
  from single_client import client
  from r2p1d_loader import loader
  from r2p1d_runner import runner

  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--beta',
                      help='Mean event interval time (Poisson), milliseconds',
                      type=int, default=100)
  parser.add_argument('-g', '--gpus', help='Number of GPUs to use',
                      type=int, default=1)
  parser.add_argument('-r', '--replicas_per_gpu',
                      help='Number of replicas per GPU', type=int, default=1)
  parser.add_argument('-bs', '--batch_size', help='Video batch size per replca',
                      type=int, default=1)
  parser.add_argument('-v', '--videos', help='Total number of videos to run',
                      type=int, default=2000)
  args = parser.parse_args()
  print('Args:', args)

  num_runners = args.gpus * args.replicas_per_gpu

  # barrier to ensure all processes start at the same time
  sta_bar_semaphore = Semaphore(0)
  sta_bar_value = Value('i', 0)
  # one client + one loader + one main process (this one) = 3
  # this needs to be changed if there are multiple clients or loaders
  sta_bar_total = num_runners + 3

  # barrier to ensure all processes finish at the same time
  fin_bar_semaphore = Semaphore(0)
  fin_bar_value = Value('i', 0)
  fin_bar_total = num_runners + 3

  # queue between client and loader
  filename_queue = SimpleQueue()
  # queue between loader and runner
  data_queue = SimpleQueue()

  pclient = Process(target=client,
                    args=(filename_queue, args.beta, args.videos,
                          sta_bar_semaphore, sta_bar_value, sta_bar_total,
                          fin_bar_semaphore, fin_bar_value, fin_bar_total))
  ploader = Process(target=loader,
                    args=(filename_queue, data_queue, num_runners,
                          sta_bar_semaphore, sta_bar_value, sta_bar_total,
                          fin_bar_semaphore, fin_bar_value, fin_bar_total))
  prunners = []
  for g in range(args.gpus):
    for r in range(args.replicas_per_gpu):
      prunners.append(Process(target=runner,
                              args=(data_queue,
                                    args.beta, args.gpus, g,
                                    args.replicas_per_gpu, r, args.batch_size,
                                    sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                    fin_bar_semaphore, fin_bar_value, fin_bar_total)))


  for p in [pclient, ploader] + prunners:
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
  tstart = time.time()
  print('START! %f' % tstart)

  with fin_bar_value.get_lock():
    fin_bar_value.value += 1
  if fin_bar_value.value == fin_bar_total:
    fin_bar_semaphore.release()
  fin_bar_semaphore.acquire()
  fin_bar_semaphore.release()

  tend = time.time()
  print('FINISH! %f' % tend)
  total_time = tend - tstart
  print('That took %f seconds' % total_time)


  print('Waiting for child processes to return...')
  for p in [pclient, ploader] + prunners:
    p.join()
  

  first_runner_log = logname(args.beta, args.gpus, 0,
                             args.replicas_per_gpu, 0, args.batch_size)
  tmp_log = 'tmp-log'
  # Append 'num_videos total_time' to the first line of the first runner log
  os.rename(first_runner_log, tmp_log)
  with open(tmp_log, 'r') as fread:
    with open(first_runner_log, 'w') as fwrite:
      fwrite.write('%d %f\n' % (args.videos, total_time))
      fwrite.write(fread.read())
  os.remove(tmp_log)
