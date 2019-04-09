class TerminationFlag:
  """An enum class for representing various termination states."""
  UNSET = -1
  TARGET_NUM_VIDEOS_REACHED = 0
  FILENAME_QUEUE_FULL = 1
  FRAME_QUEUE_FULL = 2
