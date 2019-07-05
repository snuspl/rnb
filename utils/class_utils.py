import sys
def get_class(module_path, kwargs):
    delimiter_idx = module_path.rfind('.')
    module_path = module_path[:delimiter_idx]
    class_name = module_path[delimiter_idx+1:]
    module = __import__(module_path, fromlist=(class_name))
    return getattr(module, class_name)