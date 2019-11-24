"""Ragged utility operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def pre_batch_ragged(tensor):
    return tf.RaggedTensor.from_tensor(tf.expand_dims(tensor, axis=0))


def post_batch_ragged(rt):
    return tf.RaggedTensor.from_nested_row_splits(rt.flat_values,
                                                  rt.nested_row_splits[1:])
