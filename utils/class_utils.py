def load_class(full_class_path):
  """Loads a class object from a full path to the class."""
  delimiter_idx = full_class_path.rfind('.')
  module_path = full_class_path[:delimiter_idx]
  class_name = full_class_path[delimiter_idx+1:]

  module = __import__(module_path, fromlist=(class_name))
  return getattr(module, class_name)
