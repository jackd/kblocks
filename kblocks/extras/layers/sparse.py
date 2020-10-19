import functools
from typing import List, Tuple, Union

import tensorflow as tf

from composite_layers import utils
from kblocks.ops import sparse as sparse_ops


@functools.wraps(sparse_ops.block_diagonalize_sparse)
def block_diagonalize_sparse(sparse_indices, dense_shape):
    return utils.wrap(sparse_ops.block_diagonalize_sparse, sparse_indices, dense_shape)


@functools.wraps(sparse_ops.apply_offset)
def apply_offset(
    batch_index: tf.Tensor, other_index: tf.Tensor, offset: Union[int, tf.Tensor]
):
    return utils.wrap(sparse_ops.apply_offset, batch_index, other_index, offset)


@functools.wraps(sparse_ops.block_diagonalize_sparse_general)
def block_diagonalize_sparse_general(sparse_indices, *offsets):
    return utils.wrap(
        sparse_ops.block_diagonalize_sparse_general, sparse_indices, *offsets
    )


@functools.wraps(sparse_ops.ragged_to_sparse_indices)
def ragged_to_sparse_indices(
    rt: tf.RaggedTensor, offset: tf.Tensor, dtype=None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return utils.wrap(sparse_ops.ragged_to_sparse_indices, rt, offset, dtype=dtype)


@functools.wraps(sparse_ops.unstack)
def unstack(
    st: tf.SparseTensor, axis: int = 0, num_partitions=None
) -> List[tf.SparseTensor]:
    return utils.wrap(sparse_ops.unstack, st, axis=axis, num_partitions=num_partitions)


@functools.wraps(sparse_ops.remove_dim)
def remove_dim(st: tf.SparseTensor, axis: int = 0) -> tf.SparseTensor:
    return utils.wrap(sparse_ops.remove_dim, st, axis=axis)


@functools.wraps(sparse_ops.remove_leading_dim)
def remove_leading_dim(st: tf.SparseTensor) -> tf.SparseTensor:
    return utils.wrap(sparse_ops.remove_leading_dim, st)
