from typing import Callable

import gin
import tensorflow as tf

from kblocks.scope import Scope
from kblocks.tensor_dict import TensorDict

K = tf.keras.backend


@gin.configurable(module="kb.extras.callbacks")
class ValueUpdater(tf.keras.callbacks.Callback):
    def __init__(self):
        self._batch_updates = TensorDict()
        self._epoch_updates = TensorDict()

    def _update_values(self, updates: TensorDict):
        if tf.executing_eagerly():
            for v, fn in updates.items():
                value = fn()
                K.set_value(v, value)
        else:
            sess = tf.compat.v1.get_default_session()
            assert sess is not None
            ops = [v.assign(fn()) for v, fn in updates.items()]
            sess.run(ops)

    def on_batch_end(self, batch, logs=None):
        self._update_values(self._batch_updates)

    def on_epoch_end(self, epoch, logs=None):
        self._update_values(self._epoch_updates)

    def schedule_batch_update(self, variable: tf.Variable, fn: Callable[[], tf.Tensor]):
        self._batch_updates[variable] = fn
        return variable

    def schedule_epoch_update(self, variable, fn):
        self._epoch_updates[variable] = fn
        return variable

    @property
    def used(self):
        return len(self._batch_updates) > 0 or len(self._epoch_updates) > 0


scope = Scope[ValueUpdater](name="value_updater")

get_default = scope.get_default


@gin.configurable(module="kb.extras.callbacks")
def schedule_update(
    variable: tf.Variable, fn: Callable[[], tf.Tensor], freq: str = "batch"
):
    if freq == "epoch":
        return schedule_batch_update(variable, fn)
    elif freq == "batch":
        return schedule_epoch_update(variable, fn)
    else:
        raise ValueError(
            'Invalid freq {} - must be one of "epoch" or "batch"'.format(freq)
        )


@gin.configurable(module="kb.extras.callbacks")
def schedule_batch_update(variable, fn):
    return get_default().schedule_batch_udpate(variable, fn)


@gin.configurable(module="kb.extras.callbacks")
def schedule_epoch_update(variable, fn):
    return get_default().schedule_epoch_update(variable, fn)
