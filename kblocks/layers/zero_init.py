from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


@gin.configurable(module='kb.layers')
class ZeroInit(tf.keras.layers.Layer):

    def __init__(self,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        self.initializer = tf.keras.initializers.get(initializer)
        self.constraint = tf.keras.constraints.get(constraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        super(ZeroInit, self).__init__(**kwargs)

    def get_config(self):
        config = super(ZeroInit, self).get_config()
        config.update(
            dict(
                initializer=tf.keras.initializers.serialize(self.initializer),
                constraint=tf.keras.constraints.serialize(self.constraint),
                regularizer=tf.keras.regularizers.serialize(self.regularizer),
            ))
        return config

    def build(self, input_shape):
        self.scalar = self.add_weight('scalar',
                                      shape=(),
                                      dtype=self.dtype,
                                      initializer=self.initializer,
                                      constraint=self.constraint,
                                      regularizer=self.constraint)

    def call(self, inputs):
        return inputs * self.scalar
