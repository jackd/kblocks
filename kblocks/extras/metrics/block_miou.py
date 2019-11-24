from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable
import numpy as np
import tensorflow as tf


def mean_iou(cm: tf.Tensor, dtype: tf.DType = tf.float32):
    """Compute the mean intersection-over-union via the confusion matrix."""
    # based on tf.keras.metrics.MeanIoU.result()
    sum_over_row = tf.cast(tf.math.reduce_sum(cm, axis=0), dtype=dtype)
    sum_over_col = tf.cast(tf.math.reduce_sum(cm, axis=1), dtype=dtype)
    true_positives = tf.cast(tf.linalg.diag_part(cm), dtype=dtype)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.math.reduce_sum(
        tf.cast(tf.not_equal(denominator, 0), dtype=dtype))

    iou = tf.math.divide_no_nan(true_positives, denominator)

    return tf.math.divide_no_nan(tf.reduce_sum(iou, name='mean_iou'),
                                 num_valid_entries)


class BlockMeanIoU(tf.keras.metrics.MeanIoU):
    """
    Calculate MeanIoU averaged over blocks.

    Useful for, say, multi-class semantic segmentation, where each class
    has its own difference semantic labels. Each classes semantic labels are
    assumed to be contiguous, so we calculate the mean IoU over each block then
    average across all blocks.

    Args:
        row_splits: iterable of ints
    """

    def __init__(self, row_splits: Iterable[int], name=None, dtype=None):
        self.row_splits = tuple(row_splits)
        self.num_blocks = len(self.row_splits) - 1
        super(BlockMeanIoU, self).__init__(num_classes=self.row_splits[-1],
                                           name=name,
                                           dtype=dtype)

    def result(self):
        out = []
        for i in range(self.num_blocks):
            start, end = self.row_splits[i:i + 2]
            block_cm = self.total_cm[start:end, start:end]
            out.append(mean_iou(block_cm, self.dtype))
        out.append(np.mean(out))
        out.append(mean_iou(self.total_cm))
        return np.array(out)

    def get_config(self):
        config = super(BlockMeanIoU, self).get_config()
        config['row_splits'] = self.row_splits
        return config
