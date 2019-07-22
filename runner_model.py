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

  def __call__(self, tensors, non_tensors, time_card):
    """Perform inference on this model with the given input.

    We purposely follow PyTorch's convention of using __call__ for inference.
    The first input parameter is a tuple of tensors. Note that even if there is
    only one tensor input, this parameter is still a tuple and not a standalone
    tensor object. In that case, you can extract the single tensor simply by
    doing `tensor = tensors[0]`. Moreover, this tuple is set to None if no
    tensor output has been provided by the previous step.

    The second input parameter is a non-tensor object. Unlike the tensor tuple,
    this parameter does not have any restrictions regarding its type.
    It could be a tuple, or a primitive string, or anything. This parameter is
    set to None if no non-tensor output has been given from the previous step.

    The third input parameter is a TimeCard object which holds various timings
    regarding this particular inference item. You are allowed to check, use, or
    even manipulate its contents in case your implementation involves any
    system-related aspects. Otherwise, it is perfectly fine to completely ignore
    this.


    This format is the same for the output. For the tensor outputs, make
    sure to return None if there is no output, and to return a tuple if there
    is at least one output. For the non-tensor output, either return an object
    in any desired format (tuples, lists, and other nested data structures are
    all allowed), or None if there is no such output. For the TimeCard object,
    return the input parameter as-is if you did not touch it, or a corresponding
    TimeCard object if you did.

    Return the outputs in the form of a tuple, e.g.,
    `return (tensor,), non_tensor, time_card`.
    """
    raise NotImplementedError
