from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin
from typing import Union, Optional


@gin.configurable(module="kb.extras.optimizers.schedules")
def cosine_annealing(step, max_value, min_value, steps_per_restart, dtype=tf.float32):
    step = tf.convert_to_tensor(step, dtype)
    max_value = tf.convert_to_tensor(max_value, dtype)
    min_value = tf.convert_to_tensor(min_value, dtype)
    steps_per_restart = tf.convert_to_tensor(steps_per_restart, dtype)
    return min_value + (max_value - min_value) / 2 * (
        1
        + tf.cos(
            np.pi * ((tf.math.mod(step, steps_per_restart)) / (steps_per_restart - 1))
        )
    )


@gin.configurable(module="kb.extras.optimizers.schedules")
class CosineAnnealing(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(
        self,
        max_learning_rate: float,
        min_learning_rate: float,
        steps_per_restart: Union[int, float],
        dtype: tf.DType = tf.float32,
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.steps_per_restart = steps_per_restart
        self.dtype = dtype
        self._max_learning_rate_t = tf.convert_to_tensor(max_learning_rate, dtype)
        self._min_learning_rate_t = tf.convert_to_tensor(min_learning_rate, dtype)
        self._steps_per_restart_t = tf.convert_to_tensor(steps_per_restart, dtype)

    def __call__(self, step):
        return cosine_annealing(
            tf.convert_to_tensor(step, self.dtype),
            self._min_learning_rate_t,
            self._max_learning_rate_t,
            self._steps_per_restart_t,
            dtype=self.dtype,
        )

    def get_config(self):
        return dict(
            min_learning_rate=self.min_learning_rate,
            max_learning_rate=self.max_learning_rate,
            steps_per_restart=self.steps_per_restart,
            dtype=self.dtype,
        )


@gin.configurable(module="kb.extras.optimizers.schedules")
class ExponentialDecayTowards(tf.keras.optimizers.schedules.ExponentialDecay):
    """Exponential decay scheduler with lower bound."""

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        asymptote=0,
        clip_value=None,
        staircase: bool = False,
        name: Optional[str] = None,
    ):
        super(ExponentialDecayTowards, self).__init__(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )
        self.asymptote = asymptote
        self.clip_value = clip_value

    def __call__(self, step):
        return exponential_decay_towards(
            step,
            initial_value=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            asymptote=self.asymptote,
            clip_value=self.clip_value,
            staircase=self.staircase,
            impl=tf,
        )

    def get_config(self):
        config = super(ExponentialDecayTowards, self).get_config()
        config["asyptote"] = self.asymptote
        config["clip_value"] = self.clip_value
        return config


@gin.configurable(module="kb.extras.optimizers.schedules")
def exponential_decay(
    step,
    initial_value,
    decay_steps,
    decay_rate,
    min_value=None,
    staircase=False,
    impl=tf,
):
    """
    Exponential decay schedule.

    Args:
        step: primary input
        initial_value: value when step = 0
        decay_steps: number of steps for each full decay
        decay_rate: rate of decay per `decay_steps`
        min_value: minimum value (or None for no clipping)
        staircase: if True, the floor of steps / decay_steps is used
        impl: anything with a 'floor' and 'maximum' function, e.g. np or tf

    Returns:
        possibly clipped exponentially decayed value.
    """
    step = tf.cast(step, tf.float32)
    initial_value = tf.cast(initial_value, tf.float32)
    decay_steps = tf.cast(decay_steps, tf.float32)
    decay_rate = tf.cast(decay_rate, tf.float32)
    exponent = step / decay_steps
    if staircase:
        exponent = impl.floor(exponent)
    value = initial_value * decay_rate ** exponent
    if min_value is not None:
        min_value = tf.convert_to_tensor(min_value, tf.float32)
        value = impl.maximum(value, min_value)
    return value


@gin.configurable(module="kb.extras.optimizers.schedules")
def exponential_decay_towards(
    step,
    initial_value,
    decay_steps,
    decay_rate,
    asymptote=1.0,
    clip_value=None,
    staircase=False,
    impl=tf,
):
    """
    Return exponential approaching the given asymptote.

    See expoential_decay.
    """
    kwargs = dict(
        decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase, impl=impl
    )
    # raise Exception()
    if asymptote > initial_value:
        return asymptote - exponential_decay(
            step,
            asymptote - initial_value,
            min_value=None if clip_value is None else asymptote - clip_value,
            **kwargs
        )
    else:
        return asymptote + exponential_decay(
            step,
            initial_value - asymptote,
            min_value=None if clip_value is None else clip_value - asymptote,
            **kwargs
        )
