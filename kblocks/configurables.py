from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


@gin.configurable(module="tf.ops")
def assign(variable: tf.Variable, value, use_locking=False, name=None, read_value=True):
    return variable.assign(
        value, use_locking=use_locking, name=name, read_value=read_value
    )


Variable = gin.external_configurable(tf.Variable, module="tf")
