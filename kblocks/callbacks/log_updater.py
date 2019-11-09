from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
K = tf.keras.backend


@gin.configurable(module='kb.callbacks')
class LogUpdater(tf.keras.callbacks.Callback):

    def __init__(self):
        self._batch_logs = {}
        self._epoch_logs = {}

    _stack = []

    def __enter__(self):
        LogUpdater._stack.append(self)

    def __exit__(self, type, value, traceback):
        out = LogUpdater._stack.pop()
        assert (out is self)

    @classmethod
    def get_default(cls):
        if len(cls._stack) == 0:
            raise ValueError(
                'No `LogUpdater` scopes active - must be used in `with` block')
        return cls._stack[-1]

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            for k, v in self._batch_logs.items():
                v = K.get_value(v)
                logs[k] = v

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k, v in self._epoch_logs.items():
                logs[k] = K.get_value(v)

    def _assert_not_present(self, key):
        if key in self._batch_logs or self._epoch_logs:
            raise KeyError('key {} already exists'.format(key))

    def log_each_batch(self, key, value):
        self._assert_not_present(key)
        self._batch_logs[key] = value

    def log_each_epoch(self, key, value):
        self._assert_not_present(key)
        self._epoch_logs[key] = value


def get_default():
    return LogUpdater.get_default()


def log_each_batch(key, value):
    get_default().log_each_batch(key, value)


def log_each_epoch(key, value):
    get_default().log_each_epoch(key, value)
