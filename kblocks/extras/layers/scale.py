from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


@gin.configurable(module='kb.extras.layers')
class Scale(tf.keras.layers.Layer):

    def __init__(self,
                 initializer='ones',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        self.initializer = tf.keras.initializers.get(initializer)
        self.constraint = tf.keras.constraints.get(constraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        super(Scale, self).__init__(**kwargs)

    def get_config(self):
        config = super(Scale, self).get_config()
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


@gin.configurable(module='kb.extras.layers')
def ZeroInit(initializer='zeros', regularizer=None, constraint=None, **kwargs):
    """
    Just a Scale with different default initializer.

    https://openreview.net/forum?id=BJeVklHtPr&fbclid=IwAR38PoAbf6WqlyLzMNuQURUJntGITfZvFFVb3z1a4Fiumwe1IM3lmjHGsl4
    """
    return Scale(initializer=initializer,
                 regularizer=regularizer,
                 constraint=constraint,
                 **kwargs)
