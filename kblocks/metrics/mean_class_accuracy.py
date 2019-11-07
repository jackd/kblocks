from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf
K = tf.keras.backend


@gin.configurable(module='kb.metrics')
class MeanClassAccuracy(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name='mean_class_acc', dtype=tf.float32):
        super(MeanClassAccuracy, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        shape = (num_classes,)
        self.total = self.add_weight('total',
                                     shape=shape,
                                     initializer=tf.zeros_initializer,
                                     dtype=self.dtype)
        self.correct = self.add_weight('correct',
                                       shape=shape,
                                       initializer=tf.zeros_initializer,
                                       dtype=self.dtype)

    def get_config(self):
        config = super(MeanClassAccuracy, self).get_config()
        config['num_classes'] = self.num_classes
        return self.num_classes

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype))
                           for v in self.variables])

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]
        if num_classes != self.num_classes:
            raise ValueError(
                'Expected logit/prob predictions with {} num_classes, '
                'but y_pred has shape {}'.format(self.num_classes,
                                                 y_pred.shape))

        if y_true.shape.ndims == y_pred.shape.ndims:
            y_true = tf.reshape(y_true, tf.shape(y_pred)[:-1])
        if not y_true.dtype.is_integer:
            y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        correct = tf.equal(y_true, y_pred)
        if sample_weight is None:
            sample_weight = tf.ones(shape=tf.shape(y_true), dtype=self.dtype)
        total_inc = tf.math.unsorted_segment_sum(sample_weight, y_true,
                                                 num_classes)
        correct_inc = tf.math.unsorted_segment_sum(
            tf.where(correct, sample_weight, tf.zeros_like(sample_weight)),
            y_true, num_classes)
        # total_update = K.update_add(self.total, total_inc)
        # correct_update = K.update_add(self.correct, correct_inc)
        # return total_update, correct_update
        self.total.assign_add(total_inc)
        self.correct.assign_add(correct_inc)

    def result(self):
        valid = tf.greater(self.total, 0)
        correct = tf.boolean_mask(self.correct, valid)
        total = tf.boolean_mask(self.total, valid)
        return tf.reduce_mean(correct / total)
