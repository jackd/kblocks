from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union, Iterable
import tensorflow as tf
from kblocks.tf_typing import Dimension


def flatten_leading_dims(x: Union[tf.Tensor, tf.RaggedTensor],
                         num_dims: int = 2,
                         leading_dim: Dimension = -1):
    if num_dims == 1:
        return x
    if num_dims < 1:
        raise ValueError(
            'num_dims must be non-negative, got {}'.format(num_dims))
    if isinstance(x, tf.RaggedTensor):
        return flatten_leading_dims(x.values, num_dims - 1)

    shape = x.shape
    if num_dims > shape.ndims:
        raise ValueError('num_dims {} invalid for x with shape {}'.format(
            num_dims, x.shape))
    shape = shape[num_dims:]
    if any(s is None for s in shape):
        dynamic_shape = tf.unstack(tf.shape(x)[num_dims:])
        shape = tuple(
            d if s is None else s for s, d in zip(shape, dynamic_shape))
    return tf.reshape(x, (leading_dim, *shape))


def reshape_leading_dim(x: tf.Tensor, dims: Iterable[Dimension]):
    """
    Reshape the leading dimensions of x.

    Example:
    ```python
    x = tf.random.uniform(shape=(10, 3))
    reshape_leading_dim(x, (2, 5)).shape == (2, 5, 3)
    reshape_leading_dim(x, (5, -1)).shape == (5, 2, 3)
    ```

    Args:
        x: non-scalar tensor (i.e. rank >= 1).
        dims: ints or int scalar tensors.

    Returns:
        tensor with same values as `x` but with shape `dims + x.shape[1:]`.

    See also: flatten_leading_dims, as_batched
    """
    if x.shape.ndims == 0:
        raise ValueError('Cannot reshape leading dims of a scalar')
    dims_tup = tuple(dims)
    num_unknown = sum(
        1 if isinstance(d, int) and d == -1 else 0 for d in dims_tup)
    if num_unknown > 1:
        raise ValueError(
            'At most one of dims can be -1, got {}'.format(num_unknown))

    shape = tuple(x.shape[1:])
    if any(s is None for s in shape):
        dynamic_shape = tf.unstack(tf.shape(x)[1:])
        shape = tuple(
            d if s is None else s for s, d in zip(shape, dynamic_shape))
    return tf.reshape(x, dims_tup + shape)


def as_batched(x, batch_size: Dimension = -1, element_size: Dimension = -1):
    if element_size == -1 and batch_size == -1:
        raise ValueError(
            'At most one of `batch_size` and `element_size` can be -1')
    return reshape_leading_dim(x, (batch_size, element_size))
