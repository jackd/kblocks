"""Ragged utility operations."""
from typing import Optional

import tensorflow as tf

from kblocks.tf_typing import Dimension


def lengths_to_splits(row_lengths: tf.Tensor) -> tf.Tensor:
    assert len(row_lengths.shape) == 1
    out = tf.concat(
        [tf.zeros((1,), dtype=row_lengths.dtype), tf.math.cumsum(row_lengths)], axis=0
    )
    if row_lengths.shape[0] is not None:
        out.set_shape((row_lengths.shape[0] + 1,))
    return out


def ids_to_lengths(rowids: tf.Tensor, nrows: Optional[Dimension] = None):
    rowids_32 = rowids if rowids.dtype == tf.int32 else tf.cast(rowids, tf.int32)
    if nrows is not None:
        nrows = (
            nrows
            if isinstance(nrows, int) or nrows.dtype == tf.int32
            else tf.cast(nrows, dtype=tf.int32)
        )
    out = tf.math.bincount(
        rowids_32, minlength=nrows, maxlength=nrows, dtype=rowids.dtype
    )
    nrows = tf.get_static_value(nrows)
    if isinstance(nrows, int):
        out.set_shape((nrows,))
    return out


def splits_to_lengths(row_splits: tf.Tensor) -> tf.Tensor:
    return row_splits[1:] - row_splits[:-1]


def lengths_to_ids(row_lengths: tf.Tensor, dtype=tf.int64) -> tf.Tensor:
    row_lengths = tf.convert_to_tensor(row_lengths, dtype)
    return tf.repeat(
        tf.range(tf.size(row_lengths, out_type=row_lengths.dtype)), row_lengths, axis=0
    )


def mask_to_lengths(mask: tf.Tensor, dtype=tf.int64) -> tf.Tensor:
    """
    Convert boolean mask to row lengths of rank 1 less.

    Assumes each row of masks is made up a single block of `True` values
    followed by a single block of `False` values.
    """
    if len(mask.shape) == 0:
        raise ValueError("mask cannot be a scalar")
    if mask.dtype != tf.bool:
        raise ValueError(
            "mask must have dtype bool but has dtype {}".format(mask.dtype)
        )
    return tf.math.count_nonzero(mask, axis=-1, dtype=dtype)
