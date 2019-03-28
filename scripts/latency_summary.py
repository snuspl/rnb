"""Sample script for demonstrating how to use parse_utils.

Loads all data from the logs directory and calculates the mean latency
for each job, grouped by the number of GPUs and mean event interval time.
Note: this script assumes that logs for #GPUs=1,2,3,4,5,6 are available.
"""
from parse_utils import get_data_from_all_logs

_, df = get_data_from_all_logs()

# filter out items s.t. num_replicas_per_gpu != 1
df = df[df.num_replicas_per_gpu == 1]

# filter out items s.t. batch_size != 1
df = df[df.batch_size == 1]

# remove irrelevant columns
df = df.drop(['num_replicas_per_gpu',
                            'batch_size',
                            'num_videos'], axis=1)

# calculate latencies
df['filename_queue_wait'] = (df['time_loader_start'] - df['time_enqueue_filename']) * 1000
df['frame_extraction'] = (df['time_enqueue_frames'] - df['time_loader_start']) * 1000
df['frame_queue_wait'] = (df['time_runner_start'] - df['time_enqueue_frames']) * 1000
df['gpu_comm'] = (df['time_inference_start'] - df['time_runner_start']) * 1000
df['neural_net'] = (df['time_inference_end'] - df['time_inference_start']) * 1000

# remove all timings
df = df.drop([c for c in df.columns if c.startswith('time')], axis=1)

# aggregate by num_gpus and mean_interval
groupby_keys = ['num_gpus', 'mean_interval']
values_to_aggregate = ['filename_queue_wait', 'frame_extraction', 'frame_queue_wait', 'gpu_comm', 'neural_net']
# calculate mean
df = df.groupby(groupby_keys, as_index=False)[values_to_aggregate].mean()



import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(nrows=2, ncols=3)

for i in range(2):
  for j in range(3):
    k = i * 3 + j + 1
    t_df = df[df.num_gpus == k].drop('num_gpus', axis=1)
    ax = t_df.plot.bar(ax=axes[i, j], x='mean_interval', stacked=True)

    if j == 1 and i == 1:
      ax.set_xlabel('Mean event interval time (ms)')
    else:
      ax.set_xlabel('')

    if j == 0:
      ax.set_ylabel('Latency (ms)')
    if k != 1:
      ax.get_legend().remove()
    ax.set_title('%d GPUs' % k)
    ax.set_ylim(0, 150)


plt.show()
