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
    """Returns the expected shape of the input tensor to this model."""
    raise NotImplementedError

  def __call__(self, context):
    """Perform inference on this model with the given context.

    We purposely follow PyTorch's convention of using __call__ for inference.
    """
    raise NotImplementedError
