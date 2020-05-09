import tensorflow as tf


class Printer(tf.keras.layers.Layer):
    def __init__(self, targets_fn, summarize=None, **kwargs):
        self._targets_fn = targets_fn
        self._summarize = summarize
        super(Printer, self).__init__(**kwargs)

    def call(self, inputs):
        with tf.control_dependencies(
            [tf.print(self._targets_fn(inputs), summarize=self._summarize)]
        ):
            return tf.nest.map_structure(tf.identity, inputs)

    def get_config(self):
        config = super(Printer, self).get_config()
        config["targets_fn"] = self._targets_fn
        config["summarize"] = self._summarize
        return config
