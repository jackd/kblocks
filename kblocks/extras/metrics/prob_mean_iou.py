from __future__ import absolute_import, division, print_function

from typing import Optional

import gin
import tensorflow as tf


@gin.configurable(module="kb.metrics")
class ProbMeanIoU(tf.keras.metrics.MeanIoU):
    """tf.keras.metrics.MeanIoU wrapper that takes probabilities/logits."""

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(ProbMeanIoU, self).update_state(y_true, y_pred, sample_weight)
