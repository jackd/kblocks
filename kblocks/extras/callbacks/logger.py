from typing import Callable

import gin
import tensorflow as tf
import yaml
from absl import logging

from kblocks.serialize import register_serializable


class PrintLogger(tf.keras.callbacks.Callback):
    def __init__(self, print_fn: Callable[[str], None] = print):
        self._print_fn = print_fn
        super().__init__()

    def on_train_begin(self, logs=None):
        self.model.summary(print_fn=self._print_fn)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and len(logs) > 0:
            lines = ["Finished epoch {}".format(epoch)]
            keys = sorted(logs)
            max_len = max(len(k) for k in keys)
            for k in keys:
                lines.append("{}: {}".format(k.ljust(max_len + 1), logs[k]))
            self._print_fn("\n".join(lines))


@gin.configurable(module="kb.callbacks")
@register_serializable
class AbslLogger(PrintLogger):
    def __init__(self):
        super().__init__(logging.info)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):  # pylint: disable=no-self-use
        return {}


@gin.configurable(module="kb.callbacks")
@register_serializable
class YamlLogger(tf.keras.callbacks.Callback):
    """
    Similar to tf.keras.callbacks.CSVLogger but logs to epoch -> logs yaml.

    Note unlike CSVLogger, this allows logging of logs with non-uniform logs, e.g. when
    `validation_freq != 1` in `tf.keras.Model.fit`.
    """

    def __init__(self, filename: str, append: bool = False):
        self._filename = filename
        self._append = append
        self._file = None
        super().__init__()

    def get_config(self):
        return dict(filename=self._filename, append=self._append)

    def on_train_begin(self, logs=None):
        self._file = open(self._filename, "a" if self._append else "w")

    def on_epoch_end(self, epoch, logs=None):
        self._file.write(yaml.dump({epoch: logs or {}}))

    def on_train_end(self, logs=None):
        self._file.close()


@gin.configurable(module="kb.callbacks")
@register_serializable
class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = tf.keras.backend.get_value(self.model.optimizer.lr)

    def get_config(self):  # pylint: disable=no-self-use
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
