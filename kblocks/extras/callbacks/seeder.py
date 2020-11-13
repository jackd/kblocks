import abc

import gin
import tensorflow as tf

from kblocks.serialize import register_serializable


class Seeder(tf.keras.callbacks.Callback):
    def __init__(self, seed_offset: int = 0):
        self._seed_offset = seed_offset
        super().__init__()

    def get_config(self):
        return dict(seed_offset=self._seed_offset)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def on_epoch_begin(self, epoch, logs=None):
        self._set_seed(self._seed_offset + epoch)

    @abc.abstractmethod
    def _set_seed(self, seed: int):
        raise NotImplementedError("Abstract method")


@gin.configurable(module="kb.callbacks")
@register_serializable
class GlobalSeeder(Seeder):
    """
    `tf.keras.callbacks.Callback` that sets the tf random seed each epoch.

    This allows for deterministic results from models using random ops, e.g. dropout,
    even when restarting training from a loaded checkpoint.
    """

    def _set_seed(self, seed: int):
        tf.random.set_seed(seed)


@gin.configurable(module="kb.callbacks")
@register_serializable
class GeneratorSeeder(Seeder):
    """
    `tf.keras.callbacks.Callback` that resets a `tf.random.Generator` each epoch.

    This allows for deterministic results from models using the `Generator`'s methods.
    """

    def __init__(self, rng: tf.random.Generator, seed_offset: int = 0):
        self._rng = rng
        super().__init__(seed_offset=seed_offset)

    def _set_seed(self, seed: int):
        self._rng.reset_from_seed(seed)
