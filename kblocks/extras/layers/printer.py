from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Printer(tf.keras.layers.Layer):

    def __init__(self, targets_fn, **kwargs):
        self._targets_fn = targets_fn
        super(Printer, self).__init__(**kwargs)

    def call(self, inputs):
        with tf.control_dependencies([tf.print(self._targets_fn(inputs))]):
            return tf.nest.map_structure(tf.identity, inputs)
