from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_util import (
    repeat as _repeat,
)  # pylint: disable=import-error

# slightly convoluted repeat op instead of just importing, but silences pyright


def repeat(data: tf.Tensor, repeats: tf.Tensor, axis: int, name: Optional[str] = None):
    return _repeat(data, repeats, axis, name=name)


repeat.__doc__ = _repeat.__doc__

__all__ = ["repeat"]
