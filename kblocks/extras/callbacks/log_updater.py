from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from kblocks.tf_typing import TensorOrVariable
from kblocks.scope import Scope

K = tf.keras.backend


def _update_logs(logs, var_logs):
    if tf.executing_eagerly():
        for k, v in var_logs.items():
            v = K.get_value(v)
            logs[k] = v
    else:
        sess = tf.compat.v1.get_default_session()
        var_logs = sess.run(var_logs)
        for k, v in var_logs.items():
            logs[k] = v


@gin.configurable(module="kb.extras.callbacks")
class LogUpdater(tf.keras.callbacks.Callback):
    def __init__(self):
        self._batch_logs = {}
        self._epoch_logs = {}

    def on_batch_end(self, batch, logs=None):
        if logs is not None and self._batch_logs:
            _update_logs(logs, self._batch_logs)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and self._epoch_logs:
            _update_logs(logs, self._epoch_logs)

    def _assert_not_present(self, key):
        if key in self._batch_logs or self._epoch_logs:
            raise KeyError("key {} already exists".format(key))

    def log_each_batch(self, key, value):
        self._assert_not_present(key)
        self._batch_logs[key] = value

    def log_each_epoch(self, key, value):
        self._assert_not_present(key)
        self._epoch_logs[key] = value


scope = Scope[LogUpdater](name="log_updater")

get_default = scope.get_default


@gin.configurable(module="kb.extras.callbacks")
def logged_value(key: str, value: TensorOrVariable, freq: str = "epoch"):
    if freq == "epoch":
        log_each_epoch(key, value)
    elif freq == "batch":
        log_each_batch(key, value)
    else:
        raise ValueError(
            'Invalid freq {} - must be one of "epoch" or "batch"'.format(freq)
        )
    return value


def log_each_batch(key, value):
    get_default().log_each_batch(key, value)


def log_each_epoch(key, value):
    get_default().log_each_epoch(key, value)
