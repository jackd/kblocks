from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable

import tensorflow as tf
from kblocks.ops import shape as shape_ops
from kblocks.tf_typing import Dimension
from kblocks.tf_typing import TensorLike

Lambda = tf.keras.layers.Lambda

# def Lambda(*args, **kwargs):
#     layer = tf.keras.layers.Lambda(*args, **kwargs)
#     # print(layer.name)
#     if layer.name == 'lambda_27':
#         raise Exception()
#     return layer


def size(x: tf.Tensor, out_type=tf.int64):
    return Lambda(tf.size, arguments=dict(out_type=out_type))(x)


def dimension(x: TensorLike, axis=0, out_type=tf.int64) -> tf.Tensor:
    return Lambda(shape_ops.dimension, arguments=dict(axis=axis, out_type=out_type))(x)


def flatten_leading_dims(x: TensorLike, num_dims: int = 2, leading_dim: Dimension = -1):
    assert isinstance(num_dims, int)
    if isinstance(leading_dim, int):
        return Lambda(
            shape_ops.flatten_leading_dims,
            arguments=dict(num_dims=num_dims, leading_dim=leading_dim),
        )(x)
    else:
        return Lambda(
            lambda args: shape_ops.flatten_leading_dims(
                args[0], num_dims=num_dims, leading_dim=args[1]
            )
        )([x, leading_dim])


def reshape_leading_dim(x: TensorLike, dims: Iterable[Dimension]):
    partitions = []
    args = [x]
    rest = []
    for d in dims:
        if isinstance(d, int):
            rest.append(d)
            partitions.append(1)
        else:
            args.append(d)
            partitions.append(0)

    def f(args, rest):
        x = args[0]
        args_iter = iter(args[1:])
        rest_iter = iter(rest)
        iters = [args_iter, rest_iter]
        dims = []
        for p in partitions:
            dims.append(next(iters[p]))
        return shape_ops.reshape_leading_dim(x, dims)

    return Lambda(f, arguments=dict(rest=rest))(args)


def as_batched(x: TensorLike, batch_size: Dimension = -1, element_size: Dimension = -1):
    if (
        isinstance(element_size, int)
        and element_size == -1
        and isinstance(batch_size, int)
        and batch_size == -1
    ):
        raise ValueError("At most one of `batch_size` and `element_size` can be -1")
    return reshape_leading_dim(x, (batch_size, element_size))
