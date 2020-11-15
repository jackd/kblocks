"""
Dropout implementations using `tf.random.Generator`.

Unlike `tf.keras.layers.Dropout`, these implementations create and store their own
random generator/state, meaning networks using these can be restarted part-way through
training and generate the same sequences.
"""
from typing import Optional

import gin
import tensorflow as tf

from kblocks.serialize import register_serializable


def _rng(seed: Optional[int]):
    if seed is None:
        (rng,) = tf.random.get_global_generator().split(1)
        return rng
    return tf.random.Generator.from_seed(seed)


@gin.configurable(module="kb.extras.layers")
@register_serializable
class Dropout(tf.keras.layers.Layer):
    def __init__(self, rate: float, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._rate = rate
        self._seed = seed
        self._rng = None

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def get_config(self):
        config = super().get_config()
        config.update(dict(rate=self.rate, seed=self.seed))
        return config

    def build(self, input_shape):
        if self.built:
            return
        assert self._rng is None
        self._rng = _rng(self.seed)
        super().build(input_shape)

    def _apply_training(self, inputs):
        mask = self._rng.uniform(tf.shape(inputs)) > self.rate
        return tf.where(mask, inputs / (1 - self.rate), tf.zeros_like(inputs))

    @tf.function
    def call(  # pylint: disable=arguments-differ
        self, inputs, training: Optional[bool] = None
    ):
        assert self._rng is not None
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            return self._apply_training(inputs)
        return inputs


@gin.configurable(module="kb.extras.layers")
@register_serializable
class ChannelDropout(Dropout):
    """https://arxiv.org/abs/1904.03392"""

    def _apply_training(self, inputs):
        num_channels = inputs.shape[-1]
        mask = self._rng.uniform(shape=(num_channels,)) > self.rate
        return tf.where(mask, inputs / (1 - self.rate), tf.zeros_like(inputs))
