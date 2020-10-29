"""Utilities used throughout kblocks."""
from typing import Any, Optional

import gin
import setproctitle
import tensorflow as tf


def init_optimizer_weights(model: tf.keras.Model):
    """
    Hack to ensure optimizer variables have been created.

    This is normally run on the first optimization step, but various tests save before
    running a single step. Without running this, if the optimizer state is not stored in
    a checkpoint then loading from that checkpoint won't reset the optimizer state to
    default.
    """
    optimizer = model.optimizer
    optimizer._create_slots(model.trainable_weights)  # pylint:disable=protected-access
    optimizer._create_hypers()  # pylint:disable=protected-access
    optimizer.iterations  # pylint:disable=pointless-statement


@gin.register(module="kb.utils")
def identity(x):
    return x


@gin.configurable(module="kb.utils")
def proc(title: str = "kblocks"):
    setproctitle.setproctitle(title)


class memoized_property(property):  # pylint: disable=invalid-name
    """Descriptor that mimics @property but caches output in member variable."""

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        # See https://docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            # cached = self.fget(obj)
            cached = super(memoized_property, self).__get__(obj, type)
            setattr(obj, attr, cached)
        return cached
