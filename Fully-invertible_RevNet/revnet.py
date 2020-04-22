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

  def _construct_final_block(self):    
    final_block = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Lambda(lambda x: tf.split(x, 
                                                      num_or_size_splits=[self.config.n_classes, 
                                                                          x.shape[-1] - self.config.n_classes], 
                                                      axis=1)[0])
        ],
        name="final")
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
    _, saved_hidden = self.call(inputs, training=training)
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

    return grads_all, vars_all, loss

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

  def get_x(self, y):
    ratio = np.prod(self.config.ratio)
    y = tf.reshape(y, (1, 224 // ratio, 224 // ratio, self.config.init_filters * (ratio**2)))
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
        res_x[i, j] = np.linalg.solve(self._init_block.trainable_variables[0][0, 0, :, :3].numpy().T, 
                                      y[0, i, j, :3].numpy())
    
    res_x = np.clip(res_x, 0, 1)
    plt.imshow(res_x)

    return res_x

  def get_nuisance(self, inputs):
    h = self._init_block(inputs, training=False)

    for i, block in enumerate(self._block_list):
      if self.config.ratio[i] > 1:
        h = tf.nn.space_to_depth(h, self.config.ratio[i])
      h = block(h, training=False)

    logits = tf.keras.layers.Flatten()(h)

    _, nuisance = tf.split(logits, num_or_size_splits=[self.config.n_classes, logits.shape[-1] - self.config.n_classes], axis=1)

    return nuisance
