from typing import List, Tuple, Union

import tensorflow as tf


def block_diagonalize_sparse(sparse_indices, dense_shape):
    if tf.is_tensor(dense_shape):
        dense_shape.shape.assert_has_rank(1)
        dense_shape = tf.unstack(dense_shape, axis=-1)
    batch_size, *offsets = dense_shape
    if isinstance(sparse_indices, tf.Tensor):
        sparse_indices = tf.unstack(sparse_indices, axis=-1)
    batch_dim, *indices = sparse_indices
    assert len(indices) == len(offsets)
    out_indices = []
    out_shape = []
    for offset, ind in zip(offsets, indices):
        out_size = batch_size * offset
        out_indices.append(ind + batch_dim * offset)
        out_shape.append(out_size)
    return tuple(out_indices), tuple(out_shape)


def apply_offset(
    batch_index: tf.Tensor, other_index: tf.Tensor, offset: Union[int, tf.Tensor]
):
    assert tf.is_tensor(batch_index)
    assert tf.is_tensor(other_index)
    if isinstance(offset, int) or offset.shape.ndims == 0:
        return other_index + offset * batch_index
    offset.shape.assert_has_rank(1)
    return other_index + tf.gather(offset, batch_index)


def block_diagonalize_sparse_general(sparse_indices, *offsets):
    if isinstance(sparse_indices, tf.Tensor):
        sparse_indices = tf.unstack(sparse_indices, axis=-1)
    assert len(sparse_indices) == len(offsets) + 1
    b, *rest = sparse_indices
    out = []
    for index, offset in zip(rest, offsets):
        out.append(apply_offset(b, index, offset))
    return tuple(out)


def assert_is_ragged(rt):
    assert (
        isinstance(rt, tf.RaggedTensor)
        or tf.keras.backend.is_keras_tensor(rt)
        and isinstance(rt.type_spec, tf.RaggedTensorSpec)
    )


def assert_is_sparse(st):
    assert (
        isinstance(st, tf.SparseTensor)
        or tf.keras.backend.is_keras_tensor(st)
        and isinstance(st.type_spec, tf.SparseTensorSpec)
    )


def ragged_to_sparse_indices(
    rt: tf.RaggedTensor,
    offset: tf.Tensor,
    dtype=None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    assert_is_ragged(rt)
    assert tf.is_tensor(offset)
    if dtype is None:
        dtype = rt.dtype
    assert offset.dtype == dtype
    rt.shape.assert_has_rank(3)
    assert rt.ragged_rank == 2
    b = tf.ragged.row_splits_to_segment_ids(rt.row_splits, out_type=dtype)
    i = tf.ragged.row_splits_to_segment_ids(rt.values.row_splits, out_type=dtype)
    b = tf.gather(b, i)
    j = rt.flat_values
    if j.dtype != dtype:
        j = tf.cast(j, dtype)
    j = apply_offset(b, j, offset)
    return b, i, j


def unstack(
    st: tf.SparseTensor, axis: int = 0, num_partitions=None
) -> List[tf.SparseTensor]:
    assert_is_sparse(st)
    ndims = st.dense_shape.shape[0]
    if axis < 0:
        axis = axis + ndims
    if not 0 <= axis < ndims:
        raise ValueError(
            "Invalid axis value {} for st with ndims {}".format(axis, ndims)
        )
    indices = tf.unstack(st.indices, axis=-1)
    dense_shape = st.dense_shape
    if tf.is_tensor(dense_shape):
        dense_shape = tf.unstack(dense_shape)
    else:
        dense_shape = list(dense_shape)
    if num_partitions is None:
        num_partitions = tf.get_static_value(dense_shape[axis])
        if num_partitions is None:
            raise ValueError(
                "num_partitions must be given or be convertible to a static " "value"
            )
    del dense_shape[axis]

    partitions = tf.cast(indices[axis], tf.int32)
    del indices[axis]
    indices = tf.stack(indices, axis=-1)
    indices = tf.dynamic_partition(indices, partitions, num_partitions)
    values = tf.dynamic_partition(st.values, partitions, num_partitions)
    return [tf.SparseTensor(i, v, dense_shape) for i, v in zip(indices, values)]


def remove_dim(st: tf.SparseTensor, axis: int = 0) -> tf.SparseTensor:
    assert_is_sparse(st)
    if axis == 0:
        return remove_leading_dim(st)
    indices = tf.split(st.indices, [1, -1], axis=-1)
    del indices[axis]
    indices = tf.stack(indices, axis=-1)
    dense_shape = tf.unstack(st.dense_shape)
    del dense_shape[axis]
    dense_shape = tf.stack(dense_shape, axis=0)
    return tf.SparseTensor(indices, st.values, dense_shape)


def remove_leading_dim(st: tf.SparseTensor) -> tf.SparseTensor:
    assert_is_sparse(st)
    _, indices = tf.split(st.indices, [1, -1], axis=-1)
    _, dense_shape = tf.split(st.dense_shape, [1, -1])
    return tf.SparseTensor(indices, st.values, dense_shape)
