import gin
import tensorflow as tf


@gin.configurable(module="tf.data")
def Options(**kwargs):
    options = tf.data.Options()
    for k, v in kwargs.items():
        setattr(options, k, v)
    return options
