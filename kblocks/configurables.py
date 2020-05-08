from __future__ import absolute_import, division, print_function

import gin
import tensorflow as tf


@gin.configurable(module="tf.ops")
def assign(variable: tf.Variable, value, use_locking=False, name=None, read_value=True):
    return variable.assign(
        value, use_locking=use_locking, name=name, read_value=read_value
    )


Variable = gin.external_configurable(tf.Variable, module="tf")
