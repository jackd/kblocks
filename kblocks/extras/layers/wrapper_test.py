import numpy as np
import tensorflow as tf
from kblocks.extras.layers import wrapper as wr


class WrapperTest(tf.test.TestCase):

    def test_partition(self):
        data = np.arange(5)
        partitions = [0, 0, 1, 1, 0]
        actual = wr.partition(data, partitions)
        np.testing.assert_equal(actual[0], [0, 1, 4])
        np.testing.assert_equal(actual[1], [2, 3])
        actual = wr.partition(data, partitions, 2)
        np.testing.assert_equal(actual[0], [0, 1, 4])
        np.testing.assert_equal(actual[1], [2, 3])

    def test_stitch(self):
        data = [0, 1, 4], [2, 3]
        indices = [0, 0, 1, 1, 0]
        actual = wr.stitch(indices, data)
        np.testing.assert_equal(actual, np.arange(5))


if __name__ == '__main__':
    tf.test.main()
