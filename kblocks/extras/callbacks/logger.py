import gin
import tensorflow as tf
from absl import logging

from kblocks.serialize import register_serializable


@gin.configurable(module="kb.callbacks")
@register_serializable
class AbslLogger(tf.keras.callbacks.Callback):
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):  # pylint: disable=no-self-use
        return {}

    def on_train_begin(self, logs=None):
        self.model.summary(print_fn=logging.info)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and len(logs) > 0:
            lines = ["Finished epoch {}".format(epoch)]
            keys = sorted(logs)
            max_len = max(len(k) for k in keys)
            for k in keys:
                lines.append("{}: {}".format(k.ljust(max_len + 1), logs[k]))
            logging.info("\n".join(lines))
