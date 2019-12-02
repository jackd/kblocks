from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import gin
import tensorflow as tf
from typing import Iterable, Tuple, Dict, Any, Union

Constraint = tf.keras.constraints.Constraint


@gin.configurable(module='kb.constraints')
class CompoundConstraint(Constraint):

    def __init__(self, constraints: Iterable[Constraint]):
        self._constraints = tuple(constraints)

    def __call__(self, w):
        for c in self._constraints:
            w = c(w)
        return w

    @property
    def constraints(self) -> Tuple[Constraint]:
        return self._constraints

    def get_config(self):
        return dict(constraints=[c.get_config() for c in self._constraints])


def compound_constraint(*constraints):
    if len(constraints) == 0:
        return None
    elif len(constraints) == 1:
        return constraints[0]
    else:
        return CompoundConstraint(constraints)


@gin.configurable(module='kb.constraints')
class MaxValue(Constraint):

    def __init__(self, value: Union[int, float]):
        self._value = value

    def get_config(self):
        return dict(value=self._value)

    @property
    def value(self) -> Union[int, float]:
        return self._value

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return tf.maximum(w, self._value)


@gin.configurable(module='kb.constraints')
class WeightDecay(Constraint):
    """Equivalent to regularizers.l2(decay/2) when using SGD."""

    def __init__(self, decay: float = 0.02):
        self._factor = 1 - decay
        self._decay = decay

    def get_config(self) -> Dict[str, Any]:
        return dict(decay=self._decay)

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        if self._factor != 1:
            w = self._factor * w
        return w


@gin.configurable(module='kb.constraints')
class ScheduleConstraint(Constraint):

    def __init__(self, schedule):
        if not callable(schedule):
            schedule = tf.keras.optimizers.schedules.deserialize(schedule)
        self._schedule = schedule

    def get_config(self):
        out = super(ScheduleConstraint, self).get_config()
        out['schedule'] = tf.keras.optimizers.schedules.serialize(
            self._schedule)
        return out

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return self._schedule(tf.summary.experimental.get_step())
