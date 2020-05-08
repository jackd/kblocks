import numpy as np
import tensorflow as tf
from kblocks.ragged_structure import RaggedStructure

row_splits = [0, 5, 10, 12]
row_lengths = [5, 5, 2]
value_rowids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]


class RaggedStructureTest(tf.test.TestCase):
    def row_lengths_test(self):
        rs = RaggedStructure.from_row_splits(row_splits)
        actual = self.evaluate(rs.row_lengths())
        np.testing.assert_equal(actual, row_lengths)

    def from_row_lengths_test(self):
        rs = RaggedStructure.from_row_lengths(row_lengths)
        actual = self.evaluate(rs.row_splits)
        np.testing.assert_equal(actual, row_splits)

    def value_rowids_test(self):
        rs = RaggedStructure.from_row_splits(row_splits)
        actual = self.evaluate(rs.value_rowids())
        np.testing.assert_equal(actual, value_rowids)

    def from_value_rowids_test(self):
        rs = RaggedStructure.from_value_rowids(value_rowids)
        actual = self.evaluate(rs.row_splits)
        np.testing.assert_equal(actual, row_splits)


if __name__ == "__main__":
    tf.test.main()
