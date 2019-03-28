from parse_utils import get_data_from_all_logs

_, timing_df = get_data_from_all_logs()

# filter out items s.t. num_replicas_per_gpu != 1
timing_df = timing_df[timing_df.num_replicas_per_gpu == 1]

# filter out items s.t. batch_size != 1
timing_df = timing_df[timing_df.batch_size == 1]

# remove irrelevant columns
timing_df = timing_df.drop(['num_replicas_per_gpu',
                            'batch_size',
                            'num_videos'], axis=1)

# calculate latencies
timing_df['filename_queue_wait'] = (timing_df['time_loader_start'] - timing_df['time_enqueue_filename']) * 1000
timing_df['frame_extraction'] = (timing_df['time_enqueue_frames'] - timing_df['time_loader_start']) * 1000
timing_df['frame_queue_wait'] = (timing_df['time_runner_start'] - timing_df['time_enqueue_frames']) * 1000
timing_df['gpu_comm'] = (timing_df['time_inference_start'] - timing_df['time_runner_start']) * 1000
timing_df['neural_net'] = (timing_df['time_inference_end'] - timing_df['time_inference_start']) * 1000

# remove all timings
timing_df = timing_df.drop([c for c in timing_df.columns if c.startswith('time')], axis=1)

# aggregate by num_gpus and mean_interval
groupby_keys = ['num_gpus', 'mean_interval']
values_to_aggregate = ['filename_queue_wait', 'frame_extraction', 'frame_queue_wait', 'gpu_comm', 'neural_net']
# calculate mean
timing_df = timing_df.groupby(groupby_keys, as_index=False)[values_to_aggregate].mean()



import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(nrows=2, ncols=3)

for i in range(2):
  for j in range(3):
    k = i * 3 + j + 1
    t_df = timing_df[timing_df.num_gpus == k].drop('num_gpus', axis=1)
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
