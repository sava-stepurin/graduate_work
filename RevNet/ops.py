import tensorflow as tf
import six

class HParams(object):
  def __init__(self, hparam_def=None, **kwargs):
    self._hparam_types = {}
    if hparam_def:
      self._init_from_proto(hparam_def)
      if kwargs:
        raise ValueError('hparam_def and initialization values are '
                         'mutually exclusive')
    else:
      for name, value in six.iteritems(kwargs):
        self.add_hparam(name, value)

  def add_hparam(self, name, value):
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError(
            'Multi-valued hyperparameters cannot be empty: %s' % name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    setattr(self, name, value)


def downsample(x, filters, strides, axis=1):
  """Downsample feature map with avg pooling, if filter size doesn't match."""

  def pad_strides(strides, axis=1):
    """Convert length 2 to length 4 strides.
    Needed since `tf.layers.Conv2D` uses length 2 strides, whereas operations
    such as `tf.nn.avg_pool` use length 4 strides.
    Args:
      strides: length 2 list/tuple strides for height and width
      axis: integer specifying feature dimension according to data format
    Returns:
      length 4 strides padded with 1 on batch and channel dimension
    """

    assert len(strides) == 2

    if axis == 1:
      return [1, 1, strides[0], strides[1]]
    return [1, strides[0], strides[1], 1]

  assert len(x.shape) == 4 and (axis == 1 or axis == 3)

  data_format = "NCHW" if axis == 1 else "NHWC"
  strides_ = pad_strides(strides, axis=axis)

  if strides[0] > 1:
    x = tf.nn.avg_pool(
        x, strides_, strides_, padding="VALID", data_format=data_format)

  in_filter = x.shape[axis]
  out_filter = filters

  if in_filter < out_filter:
    pad_size = [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
    if axis == 1:
      x = tf.pad(x, [[0, 0], pad_size, [0, 0], [0, 0]])
    else:
      x = tf.pad(x, [[0, 0], [0, 0], [0, 0], pad_size])
  # In case `tape.gradient(x, [x])` produces a list of `None`
  return x + 0.
