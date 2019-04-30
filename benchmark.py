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

def sanity_check(args):
  """Validate the given user arguments. 

  The function 'sanity_check' checks the arguments and terminates when an invalid state is observed. 

  The program will terminate in the following situations:
  1) The given pipeline configuration is written in an incorrect format.
  2) The values given for environment variable 'CUDA_VISIBLE_DEVICES' will be checked
     to see if valid argument is given, and the program will terminate if not so. 
  3) Here, we will regard GPUs with no process running along with no consumption in memory as 'free'.
     If a GPU has no process running, but is consuming some memory, we will regard the GPU as 'not-free', 
     and prevent users from using it. If user requires any GPU that is either
     not accessible or not free, the program will also terminate.
  """
  import json
  import os
  import sys
  from py3nvml import py3nvml

  # Case 1: Check the format of the pipeline configuration file
  try:
    with open(args.config_file_path, 'r') as f:
      pipeline = json.load(f)

    assert isinstance(pipeline, list)

    # Track which gpus are going to be used, for case 3.
    logical_gpus_to_use = set()

    for step in pipeline:
      assert isinstance(step, dict)
      assert isinstance(step['model'], str)
      assert isinstance(step['gpus'], list)
      for gpu in step['gpus']:
        assert isinstance(gpu, int)
        logical_gpus_to_use.add(gpu)

      # this limit needs to be adjusted if we expect keys other than
      # 'model' and 'gpus'
      if len(step) > 2:
        print('[WARNING] Pipeline configuration contains unused keys.')

  except Exception as err:
    print('[ERROR] Malformed pipeline configuration file. See below:')
    raise err

  # Case 2: Return 'ValueError' if any types other than integers are set for this environment variable
  logical_to_physical_gpu_idx = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
   
  # Case 3: Check whether user requires any GPU that is inaccessible or not free
  py3nvml.nvmlInit()
 
  # Find the indices of GPUs that are free
  gpu_availability = []
  for i in range(py3nvml.nvmlDeviceGetCount()):      
    handle = py3nvml.nvmlDeviceGetHandleByIndex(i) 
    memory_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
    # memory_info.used returns the consumed GPU memory usage in bits
    gpu_availability.append(memory_info.used == 0)

  # check availability of all requested gpus in the pipeline configuration
  for logical_gpu in logical_gpus_to_use:
    if logical_gpu >= len(logical_to_physical_gpu_idx):
      print('[ERROR] Pipeline configuration contains an inaccessible GPU %d. '
            'Add more GPUs to CUDA_VISIBLE_DEVICES.' % logical_gpu)
      sys.exit()
    physical_gpu = logical_to_physical_gpu_idx[logical_gpu]

    if physical_gpu >= len(gpu_availability):
      print('[ERROR] CUDA_VISIBLE_DEVICES contains a nonexistent GPU %d.'
            % physical_gpu)
      sys.exit()

    if not gpu_availability[physical_gpu]:
      print('[ERROR] GPU %d (= GPU %d in pipeline) is not free at the moment.'
            % (physical_gpu, logical_gpu))
      sys.exit()
  
  py3nvml.nvmlShutdown()
  
if __name__ == '__main__':
  # placing these imports before the if statement results in a
  # "context has already been set" RuntimeError
  from torch import multiprocessing as mp
  # https://pytorch.org/docs/stable/notes/multiprocessing.html#sharing-cuda-tensors
  mp.set_start_method('spawn')

  import argparse
  import json
  import os
  import shutil
  import time
  from arg_utils import *
  from datetime import datetime as dt
  from torch.multiprocessing import Queue, Process, Semaphore, Value

  # change these if you want to use different client/loader/runner impls
  from rnb_logging import logmeta, logroot
  from control import TerminationFlag
  from client import client
  from r2p1d_loader import loader
  from runner import runner

  parser = argparse.ArgumentParser()
  parser.add_argument('-mi', '--mean_interval_ms',
                      help='Mean event interval time (Poisson), milliseconds',
                      type=nonnegative_int, default=100)
  parser.add_argument('-b', '--batch_size', help='Video batch size per replica',
                      type=positive_int, default=1)
  parser.add_argument('-v', '--videos', help='Total number of videos to run',
                      type=positive_int, default=2000)
  parser.add_argument('-l', '--loaders', help='Number of loader processes to spawn',
                      type=positive_int, default=1)
  parser.add_argument('-qs', '--queue_size',
                      help='Maximum queue size for inter-process queues',
                      type=positive_int, default=500)
  parser.add_argument('-c', '--config_file_path',
                      help='File path of the pipeline configuration file',
                      type=str, default='config/r2p1d-whole.json')
  args = parser.parse_args()
  print('Args:', args)
  
  sanity_check(args)

  job_id = '%s-mi%d-b%d-v%d-l%d-qs%d' % (dt.today().strftime('%y%m%d_%H%M%S'),
                                         args.mean_interval_ms,
                                         args.batch_size,
                                         args.videos,
                                         args.loaders,
                                         args.queue_size)

  # do a quick pass through the pipeline to count the total number of runners
  with open(args.config_file_path, 'r') as f:
    pipeline = json.load(f)
  num_runners = sum([len(step['gpus']) for step in pipeline])
  num_runners_first_step = len(pipeline[0]['gpus'])

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
                                 args=(filename_queue, frame_queue,
                                       num_runners_first_step, l,
                                       termination_flag,
                                       sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                       fin_bar_semaphore, fin_bar_value, fin_bar_total))
                         for l in range(args.loaders)]

  # hold at least one reference for all queues
  # otherwise, a queue object may get destroyed before child processes start
  queues = [frame_queue]
  process_runner_list = []
  for step_idx, step in enumerate(pipeline):
    is_final_step = step_idx == len(pipeline) - 1

    # we don't really have to put in None here,
    # but we do anyway to avoid handling corner cases below
    queues.append(Queue(args.queue_size) if not is_final_step else None)

    model = step['model']
    gpus = step['gpus']
    replica_dict = {}
    for j, g in enumerate(gpus):
      is_first_instance = j == 0

      # check the replica index of this particular runner, for this gpu
      # if this runner is the first, then give it index 0
      replica_idx = replica_dict.get(g, 0)

      # this step should create as many markers as the number of replicas for
      # the next step (== len(pipeline[step_idx+1]['gpus']));
      # the first instance creates all markers, other instances do nothing
      num_exit_markers = len(pipeline[step_idx+1]['gpus']) \
                         if not is_final_step and is_first_instance \
                         else 0

      # we only want a single instance of the last step to print summaries
      print_summary = is_final_step and is_first_instance

      # the last two queues in `queues` are
      # the input and output queue for this step, respectively
      process_runner = Process(target=runner,
                               args=(queues[-2], queues[-1],
                                     num_exit_markers, print_summary,
                                     job_id, g, replica_idx,
                                     global_inference_counter, args.videos,
                                     termination_flag, step_idx,
                                     sta_bar_semaphore, sta_bar_value, sta_bar_total,
                                     fin_bar_semaphore, fin_bar_value, fin_bar_total,
                                     model))

      replica_dict[g] = replica_idx + 1
      process_runner_list.append(process_runner)


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

  # copy the pipeline file to the log dir of this job, for later reference
  basename = os.path.basename(args.config_file_path)
  shutil.copyfile(args.config_file_path,
                  os.path.join(logroot(job_id), basename))
