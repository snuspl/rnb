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
  def __init__(self, id):
    self.timings = OrderedDict()
    self.id = id
    self.sub_id = None
    self.diverged = None


  def record(self, key):
    """Leave a record to indicate the event `key` has occured just now."""
    self.timings[key] = time.time()


  def fork(self, sub_id):
    """Create a child clone of this TimeCard object.

    The child clone retains the same id and timings of the parent, and is
    augmented with additional information.

    The input parameter `sub_id` can be used to distinguish child clones forked
    from the same parent TimeCard. Also, an internal integer `diverged` is
    stored to keep track of when the fork was created, based on the number of
    entries in `timings`.

    The `timings` of the child is a deep copy of the `timings` of the parent,
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
    child.diverged = len(self.timings)
    return child


  @staticmethod
  def merge(time_cards):
    """Merge several TimeCards that share the same parent into one.

    A merge is only allowed for TimeCards that have the same id, the same
    timing keys, and the same diverged point.
    """
    merged_time_card = TimeCard(time_cards[0].id)
    keys = time_cards[0].timings.keys()
    diverged = time_cards[0].diverged
    for time_card in time_cards[1:]:
      if keys != time_card.timings.keys():
        raise Exception('Trying to merge TimeCards with different timing keys.'
            ' %s != %s' % (str(keys), str(time_card.timings.keys())))
      if diverged != time_card.diverged:
        raise Exception('Trying to merge TimeCards that were not forked '
                        'together. %d != %d' % (diverged, time_card.diverged))

    for key_idx, key in enumerate(keys):
      if key_idx < diverged:
        # This is before the fork happened. All TimeCards will have the same
        # timings anyway, so just copy the entries from any TimeCard.
        merged_time_card.timings[key] = time_cards[0].timings[key]
        continue

      else:
        # This is after the fork happened. For each key afterwards, append them
        # with the sub_id of the TimeCards and add them as separate entries.
        sub_ids_with_values = [(time_card.sub_id, time_card.timings[key])
                               for time_card in time_cards]
        for sub_id, value in sorted(sub_ids_with_values):
          sub_key = '%s-%s' % (key, str(sub_id))
          merged_time_card.timings[sub_key] = value

    return merged_time_card


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
    fp.write('\n')
    for tpl in zip(*self.summary.values()):
      fp.write(' '.join(map(str, tpl)))
      fp.write('\n')
