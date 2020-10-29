import numpy as np
import tensorflow as tf

from kblocks.framework.sources import core


def assert_all_equal(struct0, struct1):
    tf.nest.map_structure(np.testing.assert_equal, struct0, struct1)


class DataSourceTest(tf.test.TestCase):
    def test_tfds_reproducible(self):
        """Ensure data from TfdsSource is reproducible."""

        def get_data(split):
            source = core.TfdsSource("mnist")
            return tuple(
                (x.numpy(), label.numpy()) for x, label in source.get_dataset(split)
            )

        train0 = get_data("train")
        train1 = get_data("train")
        assert_all_equal(train0, train1)

        test0 = get_data("test")
        test1 = get_data("test")
        assert_all_equal(test0, test1)


if __name__ == "__main__":
    tf.test.main()
