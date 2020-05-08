from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union, Iterable
import tensorflow as tf
from kblocks.tf_typing import Dimension, TensorLike


def flatten_leading_dims(
    x: Union[tf.Tensor, tf.RaggedTensor], num_dims: int = 2, leading_dim: Dimension = -1
):
    if num_dims == 1:
        return tf.identity(x)
    if num_dims < 1:
        raise ValueError("num_dims must be non-negative, got {}".format(num_dims))
    if isinstance(x, tf.RaggedTensor):
        return flatten_leading_dims(x.values, num_dims - 1)

    shape = x.shape
    if num_dims > shape.ndims:
        raise ValueError(
            "num_dims {} invalid for x with shape {}".format(num_dims, x.shape)
        )
    shape = shape[num_dims:]
    if any(s is None for s in shape):
        dynamic_shape = tf.unstack(tf.shape(x)[num_dims:])
        shape = tuple(d if s is None else s for s, d in zip(shape, dynamic_shape))
    return tf.reshape(x, (leading_dim, *shape))


def reshape_leading_dim(
    x: Union[tf.Tensor, tf.RaggedTensor], dims: Iterable[Dimension]
):
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
        raise ValueError("Cannot reshape leading dims of a scalar")
    dims_tup = tuple(dims)
    num_unknown = sum(1 if isinstance(d, int) and d == -1 else 0 for d in dims_tup)
    if num_unknown > 1:
        raise ValueError("At most one of dims can be -1, got {}".format(num_unknown))

    if isinstance(x, tf.RaggedTensor):
        if len(dims_tup) != 2:
            raise NotImplementedError("TODO - recursive?")
        i, j = dims_tup
        if isinstance(i, int) and i == -1:
            i = x.nrows(out_type=tf.int64) // j
        elif isinstance(j, int) and j == -1:
            j = x.nrows(out_type=tf.int64) // i
        dtype = x.row_splits.dtype
        i = tf.cast(i, dtype)
        j = tf.cast(j, dtype)
        row_splits = tf.range(0, i * j + 1, j, dtype=dtype)
        return tf.RaggedTensor.from_row_splits(x, row_splits)

    shape = tuple(x.shape[1:])
    if any(s is None for s in shape):
        dynamic_shape = tf.unstack(tf.shape(x)[1:])
        shape = tuple(d if s is None else s for s, d in zip(shape, dynamic_shape))
    return tf.reshape(x, dims_tup + shape)


def dimension(x: TensorLike, axis=0, out_type=tf.int64) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.shape(x, out_type=out_type)[axis]
    elif isinstance(x, tf.RaggedTensor):
        if axis < 0:
            axis += x.shape.ndims
        assert axis >= 0
        if axis == 0:
            dim = x.nrows(out_type)
            if dim.dtype != out_type:
                dim = tf.cast(dim, out_type)
            return dim
        else:
            return dimension(x.values, axis - 1, out_type)
    elif isinstance(x, tf.SparseTensor):
        return x.dense_shape[axis]
    else:
        raise ValueError("Invalid type for x, {}".format(type(x)))


def as_batched(x, batch_size: Dimension = -1, element_size: Dimension = -1):
    if element_size == -1 and batch_size == -1:
        raise ValueError("At most one of `batch_size` and `element_size` can be -1")
    return reshape_leading_dim(x, (batch_size, element_size))
