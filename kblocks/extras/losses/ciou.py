from typing import Any, Dict, Optional

import gin
import tensorflow as tf


@gin.configurable(module="kb.extras.losses")
def continuous_binary_iou_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, from_logits: bool = True
) -> tf.Tensor:
    with tf.name_scope("continuous_binary_iou_loss"):
        if from_logits:
            y_pred = tf.nn.softmax(y_pred)
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)
        indices = tf.range(tf.size(y_true, out_type=tf.int64), dtype=tf.int64)
        indices = tf.stack((indices, y_true), axis=1)
        intersection = tf.gather_nd(y_pred, indices)
        union = tf.reduce_sum(y_pred, axis=-1) - intersection + 1.0
        iou = intersection / union
        loss = 1 - iou
    return loss


@gin.configurable(module="kb.extras.losses")
class ContinuousBinaryIouLoss(tf.keras.losses.Loss):
    def __init__(self, reduction="auto", from_logits=True, name=None):
        self._from_logits = from_logits
        super(ContinuousBinaryIouLoss, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super(ContinuousBinaryIouLoss, self).get_config()
        config["from_logits"] = self._from_logits
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return continuous_binary_iou_loss(y_true, y_pred, from_logits=self._from_logits)


def continuous_mean_iou_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    sample_weight: Optional[tf.Tensor] = None,
    from_logits: bool = True,
) -> tf.Tensor:
    with tf.name_scope("continuous_mean_iou_loss"):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        num_classes = y_pred.shape[-1]
        if y_true.shape.ndims != 1:
            y_true = tf.reshape(y_true, (-1,))
        if y_pred.shape.ndims != 2:
            y_pred = tf.reshape(y_pred, (-1, num_classes))

        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            if sample_weight.shape.ndims != 1:
                sample_weight = tf.reshape(sample_weight, (-1,))
            y_pred = y_pred * tf.expand_dims(sample_weight, axis=-1)

        y_true = tf.cast(y_true, tf.int64)  # for usage with keras

        continuous_cm = tf.math.unsorted_segment_sum(y_pred, y_true, num_classes)
        # continuous_cm = tf.scatter_nd(y_true, y_pred, shape=(num_classes,) * 2)
        intersections = tf.linalg.diag_part(continuous_cm)
        unions = (
            tf.reduce_sum(continuous_cm, axis=0) + tf.reduce_sum(continuous_cm, axis=1)
        ) - intersections
        valid = tf.greater(unions, 0)
        intersections = tf.boolean_mask(intersections, valid)
        unions = tf.boolean_mask(unions, valid)
        ious = intersections / unions
        return 1 - tf.reduce_mean(ious)


@gin.configurable(module="kb.extras.losses")
class ContinuousMeanIouLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits: bool = True, name: Optional[str] = None):
        self.from_logits = from_logits
        super(ContinuousMeanIouLoss, self).__init__(reduction="none", name=name)
        delattr(self, "reduction")

    def get_config(self) -> Dict[str, Any]:
        return dict(name=self.name, from_logits=self.from_logits)

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        loss = continuous_mean_iou_loss(
            y_true, y_pred, sample_weight=sample_weight, from_logits=self.from_logits
        )
        return loss

    def call(self, *args, **kwargs):
        raise NotImplementedError("Use __call__ instead")
