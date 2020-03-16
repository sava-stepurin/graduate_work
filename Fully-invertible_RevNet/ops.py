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

