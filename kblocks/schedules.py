from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin
from typing import Union


def cosine_annealing(step, min_value, max_value, steps_per_restart):
    return min_value + (max_value - min_value) / 2 * (1 + tf.cos(np.pi * (
        (step % steps_per_restart) / steps_per_restart)))


@gin.configurable(module='kb.schedules')
class CosineAnnealing(tf.keras.optimizers.schedules.ExponentialDecay):

    def __init__(self, min_learning_rate: float, max_learning_rate: float,
                 steps_per_restart: Union[int, float]):
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.steps_per_restart = steps_per_restart

    def __call__(self, step):
        return cosine_annealing(step, self.min_learning_rate,
                                self.max_learning_rate, self.steps_per_restart)

    def get_config(self):
        return dict(
            min_learning_rate=self.min_learning_rate,
            max_learning_rate=self.max_learning_rate,
            steps_per_restart=self.steps_per_restart,
        )


@gin.configurable(module='kb.schedules')
class ExponentialDecayTowards(tf.keras.optimizers.schedules.ExponentialDecay):
    """Exponential decay scheduler with lower bound."""

    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate,
                 asymptote=0,
                 clip_value=None,
                 staircase=False,
                 name=None):
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
        config['asyptote'] = self.asymptote
        config['clip_value'] = self.clip_value
        return config


@gin.configurable(module='kb.schedules')
def exponential_decay(step,
                      initial_value,
                      decay_steps,
                      decay_rate,
                      min_value=None,
                      staircase=False,
                      impl=tf):
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
    exponent = step / decay_steps
    if staircase:
        exponent = impl.floor(exponent)
    value = initial_value * decay_rate**exponent
    if min_value is not None:
        value = impl.maximum(value, min_value)
    return value


@gin.configurable(module='kb.schedules')
def exponential_decay_towards(step,
                              initial_value,
                              decay_steps,
                              decay_rate,
                              asymptote=1.0,
                              clip_value=None,
                              staircase=False,
                              impl=tf):
    """
    Return exponential approaching the given asymptote.

    See expoential_decay.
    """
    kwargs = dict(decay_steps=decay_steps,
                  decay_rate=decay_rate,
                  staircase=staircase,
                  impl=impl)
    if asymptote > initial_value:
        return asymptote - exponential_decay(
            step,
            asymptote - initial_value,
            min_value=None if clip_value is None else asymptote - clip_value,
            **kwargs)
    else:
        return asymptote + exponential_decay(
            step,
            initial_value - asymptote,
            min_value=None if clip_value is None else clip_value - asymptote,
            **kwargs)
