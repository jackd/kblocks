from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from kblocks.ops import ragged as ragged_ops

row_lengths = np.random.randint(low=0, high=100, size=20)
total = np.sum(row_lengths)
values = np.random.uniform(size=(total,)).astype(dtype=np.float32)


class RaggedTest(tf.test.TestCase):

    def test_lengths_to_splits(self):
        rt = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        actual = ragged_ops.lengths_to_splits(row_lengths)
        expected = rt.row_splits
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_equal(actual, expected)

    def test_ids_to_lengths(self):
        rt = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        rowids = rt.value_rowids()
        rl = ragged_ops.ids_to_lengths(rowids)
        np.testing.assert_equal(self.evaluate(rl), row_lengths)

    def test_splits_to_lengths(self):
        rt = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        rl = ragged_ops.splits_to_lengths(rt.row_splits)
        np.testing.assert_equal(self.evaluate(rl), row_lengths)

    def test_lengths_to_ids(self):
        rt = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        actual = ragged_ops.lengths_to_ids(row_lengths)
        expected = rt.value_rowids()
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_equal(actual, expected)

    def test_lengths_to_mask(self):
        actual = tf.RaggedTensor.from_row_lengths(values,
                                                  row_lengths).to_tensor() != 0
        expected = ragged_ops.lengths_to_mask(row_lengths)
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_equal(actual, expected)

    def test_mask_to_lengths(self):
        mask = tf.RaggedTensor.from_row_lengths(values,
                                                row_lengths).to_tensor() != 0
        rl = ragged_ops.mask_to_lengths(mask)
        np.testing.assert_equal(self.evaluate(rl), row_lengths)

    def test_row_max(self):
        max_length = 100
        num_segments = 10
        num_features = 13
        np.random.seed(123)
        row_lengths = np.random.randint(0, max_length, size=(num_segments,))
        total = np.sum(row_lengths)
        values = np.random.uniform(size=(total, num_features))

        actual = ragged_ops.row_max(values, row_lengths, num_segments,
                                    max_length)
        expected = tf.reduce_max(tf.RaggedTensor.from_row_lengths(
            values, row_lengths),
                                 axis=1)
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_equal(actual, expected)

    def test_segment_sum(self):
        values = tf.random.normal(shape=(100, 5), dtype=tf.float32)
        row_lengths = tf.constant([50, 30, 20])
        rt = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        segment_ids = rt.value_rowids()
        num_segments = row_lengths.shape[0]
        actual = ragged_ops.segment_sum(values, segment_ids, num_segments)
        expected = tf.reduce_sum(rt, axis=1)
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    tf.test.main()

    # RaggedTest().test_row_max()
