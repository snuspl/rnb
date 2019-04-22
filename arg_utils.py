import argparse
def positive_int(x):
  """Converts the argument to integer and checks whether the value is positive.
  """
  x = int(x)
  if x <= 0:
    raise argparse.ArgumentTypeError("Should be a positive integer but %d is given" % x)
  return x

def nonnegative_int(x):
  """Converts the argument to integer and checks whether the value is non-negative.
  """
  x = int(x)
  if x < 0:
    raise argparse.ArgumentTypeError("Should be a non-negative integer but %d is given" % x)
  return x

