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
  import py3nvml as nvml
  from py3nvml.py3nvml import *
  from datetime import datetime as dt
  from torch.multiprocessing import SimpleQueue, Process, Semaphore, Value

  # change these if you want to use different client/loader/runner impls
  from rnb_logging import logmeta
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
  args = parser.parse_args()
  print('Args:', args)
  
  free_gpus = nvml.get_free_gpus() 
  # return(list of Booleans): [True, True, True, True, True, False]
  # In this case, GPU #5 has a process running, thus returning False.
  # GPUs without any process running, but consuming 10MB of memory does not return False yet. 
  
  nvmlInit() 
  # Handling almost-free-GPUs with 10MB of memory being consumed as occupied   
  for i in range(len(free_gpus)):      
    handle = nvmlDeviceGetHandleByIndex(i) 
    info = nvmlDeviceGetMemoryInfo(handle)
    if info.used > 0:              
      # info.used returns the consumed GPU memory usage in bits
      # We will only regard GPU whose memory is not used at all as free, and the rest as occupied  
      free_gpus[i] = False            

  # Find the indices of free GPUs 
  free_gpus_index_list = [i for i,e in enumerate(free_gpus) if e] 
  
  invalid_argument = []
  for arg in vars(args).keys():
    # Expect all user-given argument to be positive. 
    if vars(args)[arg] < 0:  
      invalid_argument.append(arg) 
  
  if len(invalid_argument) > 0:
    for x in invalid_argument:
      print('[WARNING] Invalid number for %s. (%d) ' % (x, vars(args)[x]))
    
    if args.gpus > len(free_gpus):
      print('[WARNING] The current machine does not have %d GPUs.' % (args.gpus))
    
    elif args.gpus > len(free_gpus_index_list):
      print('[WARNING] The number of GPUs (%d) to use exceeds that of available free GPUs. (%d)'
             % (args.gpus, len(free_gpus_index_list)))
    
    sys.exit()

  job_id = '%s-mi%d-g%d-r%d-b%d-v%d-l%d' % (dt.today().strftime('%y%m%d_%H%M%S'),
                                            args.mean_interval_ms,
                                            args.gpus,
                                            args.replicas_per_gpu,
                                            args.batch_size,
                                            args.videos,
                                            args.loaders)

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

  # queue between client and loader
  filename_queue = SimpleQueue()
  # queue between loader and runner
  frame_queue = SimpleQueue()

  process_client = Process(target=client,
                           args=(filename_queue, args.mean_interval_ms,
                                 args.videos, args.loaders,
                                 sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                 fin_bar_semaphore, fin_bar_value, fin_bar_total))

  process_loader_list = [Process(target=loader,
                                 args=(filename_queue, frame_queue, num_runners, l,
                                       sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                       fin_bar_semaphore, fin_bar_value, fin_bar_total))
                         for l in range(args.loaders)]

  process_runner_list = []
  for g in range(args.gpus):
    for r in range(args.replicas_per_gpu):
      process_runner_list.append(Process(target=runner,
                                         args=(frame_queue,
                                               job_id, g, r,
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
    f.write('%f %f' % (time_start, time_end))
