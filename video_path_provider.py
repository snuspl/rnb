class VideoPathIterator:
  """Interface class for iterating video file paths to load."""
  def __init__(self):
    """Empty initialization method."""
    pass

  def __iter__(self):
    """Returns the iterator object.

    We highly recommend using itertools.cycle to let the benchmark processes
    the desired number of videos regardless of the number of existing
    videos in the file system.
    """
    raise NotImplementedError
