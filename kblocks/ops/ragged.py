"""Ragged utility operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable, Tuple, Optional
import tensorflow as tf
from kblocks.tf_typing import Dimension


def pre_batch_ragged(tensor: tf.Tensor) -> tf.RaggedTensor:
    return tf.RaggedTensor.from_tensor(tf.expand_dims(tensor, axis=0))


def post_batch_ragged(rt: tf.RaggedTensor) -> tf.RaggedTensor:
    return tf.RaggedTensor.from_nested_row_splits(rt.flat_values,
                                                  rt.nested_row_splits[1:])


def lengths_to_splits(row_lengths: tf.Tensor) -> tf.Tensor:
    assert (len(row_lengths.shape) == 1)
    out = tf.concat(
        [tf.zeros((1,), dtype=row_lengths.dtype),
         tf.math.cumsum(row_lengths)],
        axis=0)
    if row_lengths.shape[0] is not None:
        out.set_shape((row_lengths.shape[0] + 1,))
    return out


def ids_to_lengths(rowids: tf.Tensor, nrows: Optional[Dimension] = None):
    rowids_32 = rowids if rowids.dtype == tf.int32 else tf.cast(
        rowids, tf.int32)
    if nrows is not None:
        nrows = (nrows if isinstance(nrows, int) or nrows.dtype == tf.int32 else
                 tf.cast(nrows, dtype=tf.int32))
    out = tf.math.bincount(rowids_32,
                           minlength=nrows,
                           maxlength=nrows,
                           dtype=rowids.dtype)
    nrows = tf.get_static_value(nrows)
    if isinstance(nrows, int):
        out.set_shape((nrows,))
    return out


def splits_to_lengths(row_splits: tf.Tensor) -> tf.Tensor:
    return row_splits[1:] - row_splits[:-1]


def lengths_to_ids(row_lengths: tf.Tensor, dtype=tf.int64) -> tf.Tensor:
    from kblocks.ops import repeat
    row_lengths = tf.convert_to_tensor(row_lengths, dtype)
    return repeat(tf.range(tf.size(row_lengths, out_type=row_lengths.dtype)),
                  row_lengths,
                  axis=0)


def lengths_to_mask(row_lengths: tf.Tensor, size: Optional[Dimension] = None):
    return tf.sequence_mask(row_lengths, size)
    # if not isinstance(row_lengths, tf.Tensor):
    #     row_lengths = tf.convert_to_tensor(row_lengths,
    #                                        getattr(size, 'dtype', None))
    # if size is None:
    #     size = tf.reduce_max(row_lengths)
    # else:
    #     size = tf.convert_to_tensor(size, row_lengths.dtype)
    # size_t: tf.Tensor = size
    # shape = tf.concat([tf.shape(row_lengths, out_type=size_t.dtype), [size]],
    #                   axis=-1)
    # row_lengths = tf.expand_dims(row_lengths, axis=-1)
    # r = tf.range(size, dtype=row_lengths.dtype)
    # return tf.broadcast_to(r, shape) < row_lengths


def mask_to_lengths(mask: tf.Tensor) -> tf.Tensor:
    """
    Convert boolean mask to row lengths of rank 1 less.

    Assumes each row of masks is made up a single block of `True` values
    followed by a single block of `False` values.
    """
    if len(mask.shape) == 0:
        raise ValueError('mask cannot be a scalar')
    if mask.dtype != tf.bool:
        raise ValueError('mask must have dtype bool but has dtype {}'.format(
            mask.dtype))
    return tf.math.count_nonzero(mask, axis=-1)


def _row_reduction(reduction, values: tf.Tensor, row_lengths: tf.Tensor,
                   num_segments: Dimension, max_length: int) -> tf.Tensor:
    indices = tf.reshape(tf.range(num_segments * max_length),
                         (num_segments, max_length))
    mask = lengths_to_mask(row_lengths, max_length)
    indices = tf.boolean_mask(indices, mask)
    rest = values.shape[1:]
    indices = tf.expand_dims(indices, axis=1)
    dense_values = tf.scatter_nd(indices, values,
                                 [num_segments * max_length, *rest])
    dense_values = tf.reshape(dense_values, (num_segments, max_length, *rest))
    return reduction(dense_values, axis=1)


def row_max(values: tf.Tensor, row_lengths: tf.Tensor, num_segments: Dimension,
            max_length: int) -> tf.Tensor:
    return _row_reduction(tf.reduce_max, values, row_lengths, num_segments,
                          max_length)


def row_sum(values: tf.Tensor, row_lengths: tf.Tensor, num_segments: Dimension,
            max_length: int) -> tf.Tensor:
    return _row_reduction(tf.reduce_sum, values, row_lengths, num_segments,
                          max_length)


def segment_sum(values, segment_ids, num_segments):
    return tf.scatter_nd(tf.expand_dims(segment_ids, axis=-1), values,
                         [num_segments, *values.shape[1:]])
