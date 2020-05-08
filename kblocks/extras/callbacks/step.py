from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from typing import Any, Callable, List

import tensorflow as tf

from kblocks.scope import Scope

StepFn = Callable[[tf.Tensor], Any]


class StepFnCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self._summary_fns = OrderedDict()
        super(StepFnCallback, self).__init__()

    def register_step_fn(self, fn: StepFn):
        n = len(self._summary_fns)
        self._summary_fns[n] = fn
        return n

    def deregister_step_fn(self, id_):
        del self._summary_fns[id_]

    def on_epoch_end(self, epoch, logs=None):
        step = self.model.optimizer.iterations
        for fn in self._summary_fns.values():
            fn(step)


scope = Scope[StepFnCallback](name="step_fn_callback")
get_default = scope.get_default


def register_step_fn(fn: StepFn):
    return get_default().register_step_fn(fn)


def deregister_step_fn(self, id_):
    return get_default().deregister_step_fn(id_)
