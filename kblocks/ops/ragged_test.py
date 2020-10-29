import numpy as np
import tensorflow as tf

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
        rl = ragged_ops.ids_to_lengths(rowids, nrows=rt.nrows())
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

    def test_mask_to_lengths(self):
        mask = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_tensor() != 0
        rl = ragged_ops.mask_to_lengths(mask)
        np.testing.assert_equal(self.evaluate(rl), row_lengths)


if __name__ == "__main__":
    tf.test.main()
