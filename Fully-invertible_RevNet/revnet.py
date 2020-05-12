import functools
import operator

import matplotlib.pyplot as plt
import numpy as np
import six
import tensorflow as tf
from blocks import RevBlock

class RevNet(tf.keras.Model):
  """RevNet that depends on all the blocks."""

  def __init__(self, config):
    """Initialize RevNet with building blocks.
    Args:
      config: HParams object; specifies hyperparameters
    """
    super(RevNet, self).__init__()
    self.axis = 1 if config.data_format == "channels_first" else 3
    self.config = config

    self._init_block = self._construct_init_block()
    self._block_list = self._construct_intermediate_blocks()
    self._final_block = self._construct_final_block()

  def _construct_init_block(self):
    init_block = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=self.config.init_filters,
                kernel_size=1,
                strides=(1, 1),
                data_format=self.config.data_format,
                use_bias=False,
                padding="SAME",
                input_shape=self.config.input_shape,
                dtype=self.config.dtype),
        ],
        name="init")
    return init_block
    
  def _get_logit_mask(self, b_s):
    shape = int((np.prod(self.config.input_shape[:-1]) * self.config.init_filters) ** 0.5)
    mask = np.zeros((b_s, shape, shape), dtype=int)
    current_idx = 0
    for sum_idx in range((shape - 1) * 2 + 1):
      if current_idx == self.config.n_classes:
        break
      for i in range(sum_idx + 1):
        if i >= shape:
          break
        if sum_idx - i < shape:
          mask[:, i, sum_idx - i] = 1
          current_idx += 1
          if current_idx == self.config.n_classes:
            break
    return mask
    
  def _get_logits(self, x):
    mask = self._get_logit_mask(x.shape[0])
    logits = tf.reshape(tf.reshape(x, [-1])[mask.flatten() == 1], [x.shape[0], -1])
    return logits
    
  def _get_nuisance(self, x):
    mask = self._get_logit_mask(x.shape[0])
    nuisance = tf.reshape(tf.reshape(x, [-1])[mask.flatten() == 0], [x.shape[0], -1])
    return nuisance
    
  def _dct_2d(self, x):
    shape = x.shape
    block_size = int(shape[-1] ** 0.5)
    x = tf.reshape(tf.nn.depth_to_space(x, block_size), [shape[0], shape[1] * block_size, shape[2] * block_size])
    x = tf.signal.dct(x, norm="ortho")
    x = tf.transpose(tf.signal.dct(tf.transpose(x, perm=[0, 2, 1]), norm="ortho"), perm=[0, 2, 1])
    return x

  def _construct_final_block(self):
    ratio = np.prod(self.config.ratio)
    input_shape = (self.config.input_shape[0] // ratio, self.config.input_shape[1] // ratio, self.config.init_filters * (ratio**2))
    final_block = tf.keras.Sequential(
        [
            tf.keras.layers.Lambda(self._dct_2d),
            tf.keras.layers.Lambda(self._get_logits)
        ],
        name="final")
    
    if self.config.with_dense:
        final_block.add(tf.keras.layers.Dense(self.config.n_classes))
    return final_block

  def _construct_intermediate_blocks(self):
    filters = self.config.init_filters
    input_shape = self.config.input_shape

    # Aggregate intermediate blocks

    block_list = []
    for i in range(self.config.n_rev_blocks):
      # Compute input shape
      if self.config.data_format == "channels_first":
        w, h = input_shape[1], input_shape[2]
        input_shape = (filters * self.config.ratio[i]**2, w // self.config.ratio[i], h // self.config.ratio[i])
        filters *= self.config.ratio[i]**2
      else:
        w, h = input_shape[0], input_shape[1]
        input_shape = (w // self.config.ratio[i], h // self.config.ratio[i], filters * self.config.ratio[i]**2)
        filters *= self.config.ratio[i]**2

      # Add block
      rev_block = RevBlock(
          filters,
          input_shape,
          data_format=self.config.data_format,
          bottleneck=self.config.bottleneck,
          fused=self.config.fused,
          dtype=self.config.dtype)
      block_list.append(rev_block)

    return block_list
    
  def call(self, inputs, training=True):
    """Forward pass."""

    if training:
      saved_hidden = [inputs]

    curr = self._init_block(inputs, training=training)

    for i, block in enumerate(self._block_list):
      if self.config.ratio[i] > 1:
        curr = tf.nn.space_to_depth(curr, self.config.ratio[i])
      curr = block(curr, training=training)
      
    if training:
      saved_hidden.append(curr)

    logits = self._final_block(curr, training=training)

    return (logits, saved_hidden) if training else (logits, None)

  def compute_loss(self, logits, labels):
    """Compute cross entropy loss."""

    if self.config.dtype == tf.float32 or self.config.dtype == tf.float16:
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      cross_ent = loss_fn(y_true=labels, y_pred=logits)
    else:
      # `sparse_softmax_cross_entropy_with_logits` does not have a GPU kernel
      # for float64, int32 pairs
      labels = tf.one_hot(
          labels, depth=self.config.n_classes, axis=1, dtype=self.config.dtype)
      cross_ent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)

    return tf.reduce_mean(cross_ent)

  def compute_gradients(self, inputs, labels, training=True, l2_reg=True):
    """Manually computes gradients.
    When eager execution is enabled, this method also SILENTLY updates the
    running averages of batch normalization when `training` is set to True.
    Args:
      inputs: Image tensor, either NHWC or NCHW, conforming to `data_format`
      labels: One-hot labels for classification
      training: Use the mini-batch stats in batch norm if set to True
      l2_reg: Apply l2 regularization
    Returns:
      list of tuples each being (grad, var) for optimizer to use
    """

    # Run forward pass to record hidden states; avoid updating running averages
    vars_and_vals = self.get_moving_stats()
    logits, saved_hidden = self.call(inputs, training=training)
    self.restore_moving_stats(vars_and_vals)

    grads_all = []
    vars_all = []

    # Manually backprop through last block
    x = saved_hidden[-1]
    with tf.GradientTape() as tape:
      x = tf.identity(x)
      tape.watch(x)
      # Running stats updated below
      logits = self._final_block(x, training=training)
      loss = self.compute_loss(logits, labels)

    grads_combined = tape.gradient(loss,
                                   [x] + self._final_block.trainable_variables)
    dy, grads_ = grads_combined[0], grads_combined[1:]
    grads_all += grads_
    vars_all += self._final_block.trainable_variables

    # Manually backprop through intermediate blocks
    y = saved_hidden.pop()
    for i, block in enumerate(reversed(self._block_list)):
      y, dy, grads, vars_ = block.backward_grads_and_vars(
          y, dy, training=training)
      grads_all += grads
      vars_all += vars_

      if self.config.ratio[self.config.n_rev_blocks - 1 - i] > 1:
        y = tf.nn.depth_to_space(y, self.config.ratio[self.config.n_rev_blocks - 1 - i])
        dy = tf.nn.depth_to_space(dy, self.config.ratio[self.config.n_rev_blocks - 1 - i])

    # Manually backprop through first block
    x = saved_hidden.pop()
    assert not saved_hidden  # Cleared after backprop

    with tf.GradientTape() as tape:
      x = tf.identity(x)
      # Running stats updated below
      y = self._init_block(x, training=training)

    grads_all += tape.gradient(
        y, self._init_block.trainable_variables, output_gradients=dy)
    vars_all += self._init_block.trainable_variables

    # Apply weight decay
    if l2_reg:
      grads_all = self._apply_weight_decay(grads_all, vars_all)

    return grads_all, vars_all, loss, logits

  def _apply_weight_decay(self, grads, vars_):
    """Update gradients to reflect weight decay."""
    # Don't decay bias
    return [
        g + self.config.weight_decay * v if v.name.endswith("kernel:0") else g
        for g, v in zip(grads, vars_)
    ]
  
  def get_moving_stats(self):
    """Get moving averages of batch normalization.
    This is needed to avoid updating the running average twice in one iteration.
    Returns:
      A dictionary mapping variables for batch normalization moving averages
      to their current values.
    """
    vars_and_vals = []

    def _is_moving_var(v):
      n = v.name
      return n.endswith("moving_mean:0") or n.endswith("moving_variance:0")

    for v in filter(_is_moving_var, self.variables):
      vars_and_vals.append((v, v.read_value()))

    return vars_and_vals

  def restore_moving_stats(self, vars_and_vals):
    """Restore moving averages of batch normalization.
    This is needed to avoid updating the running average twice in one iteration.
    Args:
      vars_and_vals: The dictionary mapping variables to their previous values.
    """
    for var_, val in vars_and_vals:
      var_.assign(val)

  def get_x(self, logits, nuisance):
    # inverse last dense layer
    logits_before_dense_np = logits.numpy()[0]
    if self.config.with_dense:
        W = self._final_block.trainable_variables[0].numpy()
        b = self._final_block.trainable_variables[1].numpy()
        logits_before_dense_np = np.linalg.solve(W.T, logits.numpy()[0] - b)
        
    # make tensor with logits + nuisance
    b_s = logits.shape[0]
    mask = self._get_logit_mask(b_s)
    shape = int((np.prod(self.config.input_shape[:-1]) * self.config.init_filters) ** 0.5)
    np_res = np.zeros(b_s * shape * shape)
    np_res[mask.flatten() == 1] = tf.reshape(logits_before_dense_np, [-1])
    np_res[mask.flatten() == 0] = tf.reshape(nuisance, [-1])
    y = tf.convert_to_tensor(np_res.reshape((b_s, shape, shape)), dtype=tf.float32)
    
    # inverse dct_2d
    ratio = np.prod(self.config.ratio)
    y_shape = (logits.shape[0], self.config.input_shape[0] // ratio, self.config.input_shape[1] // ratio, self.config.init_filters * (ratio**2))
    block_size = int(y_shape[-1] ** 0.5)
    y = tf.transpose(tf.signal.idct(tf.transpose(y, perm=[0, 2, 1]), norm="ortho"), perm=[0, 2, 1])
    y = tf.signal.idct(y, norm="ortho")
    y = tf.reshape(y, [y_shape[0], y_shape[1] * block_size, y_shape[2] * block_size] + [1])
    y = tf.nn.space_to_depth(y, block_size)

    for i, block in enumerate(reversed(self._block_list)):
      res_block = block

      y1, y2 = tf.split(y, num_or_size_splits=2, axis=-1)
      z1 = y1
      gz1 = res_block.g(z1, training=False)
      x2 = y2 - gz1
      fx2 = res_block.f(x2, training=False)
      x1 = z1 - fx2
      x = tf.concat([x1, x2], axis=-1)
      y = x
      
      if self.config.ratio[self.config.n_rev_blocks - 1 - i] > 1:
        y = tf.nn.depth_to_space(y, self.config.ratio[self.config.n_rev_blocks - 1 - i])
    
    res_x = np.zeros(self.config.input_shape)
    for i in range(res_x.shape[0]):
      for j in range(res_x.shape[1]):
        res_x[i, j] = np.linalg.solve(self._init_block.trainable_variables[0][0, 0, :, :self.config.input_shape[-1]].numpy().T, 
                                      y[0, i, j, :self.config.input_shape[-1]].numpy())

    return res_x

  def get_nuisance(self, inputs):
    h = self._init_block(inputs, training=False)

    for i, block in enumerate(self._block_list):
      if self.config.ratio[i] > 1:
        h = tf.nn.space_to_depth(h, self.config.ratio[i])
      h = block(h, training=False)

    h = self._dct_2d(h)
    nuisance = self._get_nuisance(h)

    return nuisance
