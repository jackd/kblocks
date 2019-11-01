from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


@gin.configurable(module='kb.layers')
class ZeroInit(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.scalar = self.add_weight('scalar',
                                      shape=(),
                                      dtype=self.dtype,
                                      initializer=tf.keras.initializers.zeros())

    def call(self, inputs):
        return inputs * self.scalar
