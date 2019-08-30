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


def logname(job_id, g_idx, group_idx, instance_idx):
  root = logroot(job_id)
  return '%s/g%d-group%d-%d.txt' % (root, g_idx, group_idx, instance_idx)


class TimeCard:
  """Wrapper of OrderedDict for representing a list of events w/ timestamps."""
  def __init__(self, id):
    self.timings = OrderedDict()
    self.id = id
    self.sub_id = None
    self.num_parent_timings = None
    self.gpus = []


  def record(self, key):
    """Leave a record to indicate the event `key` has occured just now."""
    self.timings[key] = time.time()


  def add_gpu(self, gpu):
    """Indicate that this TimeCard has passed through a certain gpu."""
    self.gpus.append((gpu,))


  def fork(self, sub_id):
    """Create a child clone of this TimeCard object.

    The child clone retains the same id and timings of the parent, and is
    augmented with additional information.

    The input parameter `sub_id` can be used to distinguish child clones forked
    from the same parent TimeCard. Also, an internal integer
    `num_parent_timings` is stored to keep track of when the fork was created,
    based on the number of entries in `timings`.

    The instance attributes of the child are deep copies of those of the parent,
    so altering one will not affect the other.

    Note that forking from a TimeCard that was forked from yet another TimeCard
    (two-level forking) is not allowed. A merge must be performed before another
    fork is possible.
    """
    if self.sub_id is not None:
      raise Exception('Trying to fork from TimeCard(id %d, sub_id %d).' % 
                      (self.id, self.sub_id))

    child = TimeCard(self.id)
    child.timings = OrderedDict(self.timings)
    child.sub_id = sub_id
    child.num_parent_timings = len(self.timings)
    child.gpus = list(self.gpus)
    return child


  @staticmethod
  def merge(time_cards):
    """Merge several TimeCards that share the same parent into one.

    A merge is only allowed for TimeCards that have the same id, the same
    timing keys, and the same num_parent_timings.
    """
    merged_time_card = TimeCard(time_cards[0].id)
    keys = time_cards[0].timings.keys()
    num_parent_timings = time_cards[0].num_parent_timings
    for time_card in time_cards[1:]:
      if keys != time_card.timings.keys():
        raise Exception('Trying to merge TimeCards with different timing keys.'
            ' %s != %s' % (str(keys), str(time_card.timings.keys())))
      if num_parent_timings != time_card.num_parent_timings:
        raise Exception('Trying to merge TimeCards that were not forked '
            'together. %d != %d' % (num_parent_timings,
                                    time_card.num_parent_timings))

    # sort the TimeCards based on their sub_ids, for later use
    time_cards = sorted(time_cards, key=lambda time_card: time_card.sub_id)

    for key_idx, key in enumerate(keys):
      if key_idx < num_parent_timings:
        # This is before the fork happened. All TimeCards will have the same
        # timings anyway, so just copy the entries from any TimeCard.
        merged_time_card.timings[key] = time_cards[0].timings[key]

      else:
        # This is after the fork happened. For each key afterwards, append them
        # with the sub_id of the TimeCards and add them as separate entries.
        for time_card in time_cards:
          sub_key = '%s-%s' % (key, time_card.sub_id)
          merged_time_card.timings[sub_key] = time_card.timings[key]

    # merge the gpu logs
    for gpu_per_time_card in zip(*[time_card.gpus for time_card in time_cards]):
      gpu_per_time_card = tuple(gpu for tpl in gpu_per_time_card for gpu in tpl)
      # gpu_per_time_card is a tuple of gpu indices, denoting the gpus that
      # were used for a single step, e.g.,
      # (1, 1, 1) --> all TimeCards passed through gpu 1 for this step
      # (1, 2, 3) --> TimeCard 1 used gpu 1, TimeCard 2 used gpu 2, ..
      if len(set(gpu_per_time_card)) == 1:
        # all TimeCards passed through the same gpu
        # simply store that single gpu index
        merged_time_card.gpus.append((gpu_per_time_card[0],))
      else:
        # TimeCards passed through different gpus
        # store the whole tuple
        merged_time_card.gpus.append(gpu_per_time_card)

    return merged_time_card


class TimeCardSummary:
  """An aggregator class for TimeCards."""
  def __init__(self):
    self.summary = OrderedDict()
    self.gpus_per_inference = []


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

    self.gpus_per_inference.append(time_card.gpus)


  def print_summary(self, num_skips):
    """Prints a quick summary on the elapsed time between events, to stdout.

    The parameter `num_skips` can be used to indicate the number of instances
    to skip when calculating average elapsed time.
    """
    for prv, nxt in zip(self.keys[:-1], self.keys[1:]):
      if len(self.summary[prv]) <= num_skips:
        print('Not enough log entries (%d records) to print summary!' % len(self.summary[prv]))
        break

      elapsed_time = np.mean((
          np.array(self.summary[nxt][num_skips:]) -
          np.array(self.summary[prv][num_skips:])) * 1000)
      print('Average time between %s and %s: %f ms' % (prv, nxt, elapsed_time))


  def save_full_report(self, fp):
    """Logs all collected timings to the given file pointer."""
    fp.write(' '.join(self.keys))

    # this should always be true, unless there was an error in the code and
    # the experiment somehow terminated early
    if len(self.gpus_per_inference) > 0:
      for step_idx, gpu_per_time_card in enumerate(self.gpus_per_inference[0]):
        if len(gpu_per_time_card) > 1:
          # more than one gpu was used for this step
          # create separate keys for each segment
          for sub_id in range(len(gpu_per_time_card)):
            fp.write(' ')
            fp.write('gpu%d-%d' % (step_idx, sub_id))
        else:
          # only one gpu was used for this step
          fp.write(' ')
          fp.write('gpu%d' % step_idx)

    fp.write('\n')
    for tpl, gpus_per_step in zip(zip(*self.summary.values()),
                              self.gpus_per_inference):
      fp.write(' '.join(map(str, tpl)))
      for gpu_per_time_card in gpus_per_step:
        for gpu in gpu_per_time_card:
          fp.write(' %d' % gpu)
      fp.write('\n')
