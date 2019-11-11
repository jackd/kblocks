from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Callable, List
import tensorflow as tf
from collections import OrderedDict

StepFn = Callable[[tf.Tensor], Any]


class StepFnCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        self._summary_fns = OrderedDict()
        super(StepFnCallback, self).__init__()

    _stack: List['StepFnCallback'] = []

    def __enter__(self):
        StepFnCallback._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        top = StepFnCallback._stack.pop()
        assert (top is self)

    @classmethod
    def get_default(cls):
        if len(cls._stack) == 0:
            raise ValueError('No scope entered for class {}'.format(
                cls.__name__))
        return cls._stack[-1]

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


def get_default() -> StepFnCallback:
    return StepFnCallback.get_default()


def register_step_fn(fn: StepFn):
    return get_default().register_step_fn(fn)


def deregister_step_fn(self, id_):
    return get_default().deregister_step_fn(id_)
