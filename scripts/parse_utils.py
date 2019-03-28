import os
import pandas as pd
import re

def parse_conf(conf):
  """Extract arguments from log meta files.

  Example line:
  Args: Namespace(batch_size=1, gpus=2, mean_interval_ms=60, replicas_per_gpu=1, videos=2000)
  """
  pattern = re.compile(r'mean_interval_ms=(\d+)')
  mi = int(pattern.search(conf).group(1))

  pattern = re.compile(r'gpus=(\d+)')
  g = int(pattern.search(conf).group(1))

  pattern = re.compile(r'replicas_per_gpu=(\d+)')
  r = int(pattern.search(conf).group(1))

  pattern = re.compile(r'batch_size=(\d+)')
  b = int(pattern.search(conf).group(1))

  pattern = re.compile(r'videos=(\d+)')
  v = int(pattern.search(conf).group(1))

  return (mi, g, r, b, v)


def get_data(job_dir):
  """Fetch and parse all data from logs of a single job.

  The given directory is expected to contain a meta log file as well as timing
  logs for each gpu/replica, like the following example:
  $ ls logs/190327_162712-mi60-g2-r1-b1-v2000
  g0-r0.txt    g1-r0.txt    log-meta.txt

  Args:
    job_dir: the root log directory for a particular job
             e.g., logs/190327_162712-mi60-g2-r1-b1-v2000
  """
  job_inferences = {}
  for filename in os.listdir(job_dir):
    path = os.path.join(job_dir, filename)
    with open(path, 'r') as f:
      if filename == 'log-meta.txt':
        conf = f.readline()
        conf = conf.strip()
        args = parse_conf(conf)

        line = f.readline()
        time_start, time_end = tuple(map(float, line.strip().split()))

      else:
        pattern = re.compile(r'g(\d+)-r(\d+)')
        match = pattern.search(filename)
        gpu_idx = int(match.group(1))
        replica_idx = int(match.group(2))
        inferences = []

        f.readline()
        for line in f:
          inferences.append(tuple(map(float, line.strip().split())))

        job_inferences[(gpu_idx, replica_idx)] = inferences

  return args, time_start, time_end, job_inferences


def get_data_from_all_logs(log_dir='logs'):
  """Parse data for all jobs and return results as two Pandas DataFrames.

  This function reads all data from subdirectories (each subdirectory should
  represent a single job) within `log_dir` and packs them into two large
  DataFrames: a throughput DataFrame and a timing DataFrame. In case there
  are multiple jobs with the same arguments, only the most recent job is
  read; logs from old jobs are completely ignored.

  The throughput DataFrame contains the following columns:
    'mean_interval' (int, milliseconds)
    'num_gpus' (int)
    'num_replicas_per_gpu' (int)
    'batch_size' (int)
    'num_videos' (int)
    'time_start' (float, seconds)
    'time_end' (float, seconds)
    'throughput' == num_videos / (time_end - time_start) (videos/sec)
  Each row represents a single job.

  The timing DataFrame contains the following columns:
    'mean_interval' (int, milliseconds)
    'num_gpus' (int)
    'gpu_index' (int)
    'num_replicas_per_gpu' (int)
    'replica_index' (int)
    'batch_size' (int)
    'num_videos' (int)
    'time_enqueue_filename' (float, seconds)
    'time_loader_start' (float, seconds)
    'time_enqueue_frames' (float, seconds)
    'time_runner_start' (float, seconds)
    'time_inference_start' (float, seconds)
    'time_inference_end' (float, seconds)
  Each row represents the inference lifetime of a single video query. For
  example, if there are logs for 20 jobs that process 2000 videos each, then
  this DataFrame would have 20 * 2000 = 40000 rows.
  """
  args_list = []
  throughput_list = []
  timing_list = []

  for job in sorted(os.listdir(log_dir), reverse=True):
    if job.startswith('.'): continue

    path = os.path.join(log_dir, job)
    args, time_start, time_end, job_inferences = get_data(path)

    if args in args_list:
      print(args, 'has been detected again; loading the most recent data only.')
      continue
    args_list.append(args)

    mean_interval, num_gpus, num_replicas_per_gpu, \
        batch_size, num_videos = args
    throughput = num_videos / (time_end - time_start)

    throughput_list.append({
        'mean_interval': mean_interval,
        'num_gpus': num_gpus,
        'num_replicas_per_gpu': num_replicas_per_gpu,
        'batch_size': batch_size,
        'num_videos': num_videos,
        'time_start': time_start,
        'time_end': time_end,
        'throughput': throughput,
    })

    for (g_idx, r_idx), inferences in job_inferences.items():
      for (enqueue_filename, loader_start, enqueue_frames,
           runner_start, inference_start, inference_end) in inferences:
        timing_list.append({
            'mean_interval': mean_interval,
            'num_gpus': num_gpus,
            'gpu_index': g_idx,
            'num_replicas_per_gpu': num_replicas_per_gpu,
            'replica_index': r_idx,
            'batch_size': batch_size,
            'num_videos': num_videos,
            'time_enqueue_filename': enqueue_filename,
            'time_loader_start': loader_start,
            'time_enqueue_frames': enqueue_frames,
            'time_runner_start': runner_start,
            'time_inference_start': inference_start,
            'time_inference_end': inference_end,
        })

  return pd.DataFrame(throughput_list), pd.DataFrame(timing_list)
