import collections
class VideoPathIterator():
  """Interface class for iterating video file paths to load."""
  def __init__(self):
    """Empty initialization method."""
    pass

  def __iter__(self):
    """Returns the iterator object."""
    return self
    
  def __next__(self):
    """Returns the next item of the iterator."""
    pass
