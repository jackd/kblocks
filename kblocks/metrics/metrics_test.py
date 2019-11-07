from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kblocks import metrics
import numpy as np


class SchedulesTest(tf.test.TestCase):

    def test_mean_class_accuracy(self):
        metric = metrics.MeanClassAccuracy(3)
        self.assertEqual(metric.num_classes, 3)

        metric.update_state(
            tf.convert_to_tensor([0, 1, 1, 2]),
            tf.convert_to_tensor([
                [0.5, 0.2, 0.3],
                [0.2, 0.5, -0.1],
                [0.1, 0.3, 1.5],
                [0.1, 0.3, 1.5],
            ]))
        np.testing.assert_allclose(self.evaluate(metric.result()), 2.5 / 3)
        metric.update_state(
            tf.convert_to_tensor([0, 1, 1, 2]),
            tf.convert_to_tensor([
                [0.5, 0.2, 0.3],
                [0.2, 0.5, -0.1],
                [0.1, 0.3, 1.5],
                [0.1, 0.3, 1.5],
            ]), tf.convert_to_tensor([1., 1., 0.5, 1]))
        np.testing.assert_allclose(self.evaluate(metric.result()),
                                   (1 + 4. / 7 + 1) / 3)


if __name__ == '__main__':
    tf.test.main()
