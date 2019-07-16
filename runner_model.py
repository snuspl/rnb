class RunnerModel:
  """Interface class for representing models in the RnB benchmark."""
  def __init__(self, device):
    """Empty initialization method.

    We don't actually do anything with the `device` parameter, but we keep it
    to encourage child classes to place models on the correct device using the
    parameter.

    Model initialization as well as loading pretrained weights is expected
    to be done in this method. Model warm-up is NOT expected to be done here,
    and is instead done by the caller.
    """
    pass

  def input_shape(self):
    """Returns the expected shapes of the input tensors to this model.

    The return value should be a nested tuple, containing a shape tuple for each
    expected input tensor. Note that this applies even if the model expects only
    one tensor; you can create a single-item tuple by doing `(shape,)`.

    If the model does not receive any tensors, then return None. You can still
    receive any non-tensor objects the previous step passes, in __call__().
    Keep in mind that returning None and returning an empty tuple (`()`) are
    completely different. Copy-paste the previous step's output shape to be
    safe. See output_shape() for more details.
    """
    raise NotImplementedError

  @staticmethod
  def output_shape():
    """Returns the expected shape of the output tensors of this model.

    The return value should be a nested tuple, containing a shape tuple for each
    expected output tensor. Note that this applies even if the model outputs
    only one tensor; you can create a single-item tuple by doing `(shape,)`.

    If the model does not output any tensors, then return None. You are still
    allowed to output any non-tensor objects, in __call__().
    Keep in mind that returning None and returning an empty tuple (`()`) are
    completely different. For the former, the benchmark does not even bother
    creating any synchronization (multiprocessing.Event) objects for sharing
    tensors, but for the latter, the benchmark does create them.
    """
    raise NotImplementedError

  def __call__(self, input):
    """Perform inference on this model with the given input.

    We purposely follow PyTorch's convention of using __call__ for inference.
    The input parameter is a pair of tensor tuples and a non-tensor object
    (which could also be a tuple, but does not necessaily have to be), e.g.,
    ((tensor1, tensor2, tensor3), string). In case the previous step does not
    provide any tensor outputs, the tensor tuple is set to None. This is also
    true for the non-tensor object.

    Note that even if there is only one tensor input, the tensor tuple is still
    a tuple and not a standalone tensor object. In that case, one way you can
    extract the single tensor from `input` is `(tensor,), _ = input`. This is
    NOT true for the non-tensor object; the non-tensor output from the previous
    step can be literally anything.

    This tuple format is the same for the output. For the tensor outputs, make
    sure to return None if there is no output, and to return a tuple if there
    is at least one output. Also don't forget to return None for the non-tensor
    object if you don't have any non-tensor output.
    """
    raise NotImplementedError
