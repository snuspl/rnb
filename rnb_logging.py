import numpy as np
import os
import time
from collections import OrderedDict

def logroot(job_id):
  path = 'logs/%s' % job_id
  os.makedirs(path, exist_ok=True)
  return path


def logmeta(job_id):
  root = logroot(job_id)
  return '%s/log-meta.txt' % root


def logname(job_id, g_idx, r_idx):
  root = logroot(job_id)
  return '%s/g%d-r%d.txt' % (root, g_idx, r_idx)


class TimeCard:
  """Wrapper of OrderedDict for representing a list of events w/ timestamps."""
  def __init__(self):
    self.timings = OrderedDict()


  def record(self, key):
    """Leave a record to indicate the event `key` has occured just now."""
    self.timings[key] = time.time()


class TimeCardSummary:
  """An aggregator class for TimeCards."""
  def __init__(self):
    self.summary = OrderedDict()


  def register(self, time_card):
    """Stash all information stored in the given TimeCard instance.

    We assume that the type and order of events in subsequent TimeCards are
    always the same.
    """
    if len(self.summary) == 0:
      self.keys = list(time_card.timings.keys())
      for key in self.keys:
        self.summary[key] = []

    assert self.keys == list(time_card.timings.keys())

    for key, ts in time_card.timings.items():
      self.summary[key].append(ts)


  def summarize(self, num_skips):
    """Prints a quick summary on the elapsed time between events, to stdout.

    The parameter `num_skips` can be used to indicate the number of instances
    to skip when calculating average elapsed time.
    """
    for prv, nxt in zip(self.keys[:-1], self.keys[1:]):
      elapsed_time = np.mean((
          np.array(self.summary[nxt][num_skips:]) -
          np.array(self.summary[prv][num_skips:])) * 1000)
      print('Average time between %s and %s: %f ms' % (prv, nxt, elapsed_time))


  def full_report(self, fp):
    """Logs all collected timings to the given file pointer."""
    fp.write(' '.join(self.keys))
    fp.write('\n')
    for tpl in zip(*self.summary.values()):
      fp.write(' '.join(map(str, tpl)))
      fp.write('\n')
