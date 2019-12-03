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


def reshape_leading_dim(x: Union[tf.Tensor, tf.RaggedTensor],
                        dims: Iterable[Dimension]):
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

    if isinstance(x, tf.RaggedTensor):
        if len(dims_tup) != 2:
            raise NotImplementedError('TODO - recursive?')
        i, j = dims_tup
        if isinstance(i, int) and i == -1:
            i = x.nrows(out_type=tf.int64) // j
        elif isinstance(j, int) and j == -1:
            j = x.nrows(out_type=tf.int64) // i
        row_splits = tf.range(0, i * j + 1, j)
        return tf.keras.layers.Lambda(
            lambda args: tf.RaggedTensor.from_row_splits(*args))(
                [x, row_splits])

    shape = tuple(x.shape[1:])
    if any(s is None for s in shape):
        dynamic_shape = tf.unstack(tf.shape(x)[1:])
        shape = tuple(
            d if s is None else s for s, d in zip(shape, dynamic_shape))
    return tf.reshape(x, dims_tup + shape)


def _dimension(x, axis=0, out_type=tf.int64):
    return tf.shape(x, out_type=out_type)[axis]


def dimension(x, axis=0, out_type=tf.int64) -> Dimension:
    dim = x.shape[axis]
    if dim is not None:
        return dim
    if isinstance(x, tf.RaggedTensor):
        if axis < 0:
            axis += x.shape.ndims
        assert (axis >= 0)
        if axis == 0:
            return dimension(x.row_splits, 0, out_type) - 1
        else:
            return dimension(x.values, axis - 1, out_type)
    # lambda wrapper sometimes needed to avoid tf node.inputs being empty??
    return tf.keras.layers.Lambda(_dimension,
                                  arguments=dict(axis=axis,
                                                 out_type=out_type))(x)


def as_batched(x, batch_size: Dimension = -1, element_size: Dimension = -1):
    if element_size == -1 and batch_size == -1:
        raise ValueError(
            'At most one of `batch_size` and `element_size` can be -1')
    return reshape_leading_dim(x, (batch_size, element_size))
