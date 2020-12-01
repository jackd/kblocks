import os
import tempfile

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from kblocks.data import cache, snapshot
from kblocks.data.cache import save_load_cache, tfrecords_cache

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def as_array(dataset: tf.data.Dataset) -> np.ndarray:
    return np.array([el.numpy() for el in dataset])


lazy_factories = (
    cache,
    snapshot,
)

eager_factories = (
    tfrecords_cache,
    save_load_cache,
)

factories = lazy_factories + eager_factories


class CacheTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*factories)
    def test_cache_transform(self, factory):
        seed = 0
        epoch_length = 5
        rng = tf.random.Generator.from_seed(seed)
        dataset = tf.data.Dataset.range(epoch_length).map(
            lambda x: tf.cast(x, tf.float32) + rng.uniform(())
        )

        expected = as_array(dataset)
        state = rng.state.numpy()
        assert np.all(expected != as_array(dataset))

        rng.reset_from_seed(seed)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cached = dataset.apply(factory(tmp_dir))
            np.testing.assert_equal(cached.cardinality().numpy(), epoch_length)
            for _ in range(2):
                np.testing.assert_equal(as_array(cached), expected)
                np.testing.assert_equal(rng.state.numpy(), state)


if __name__ == "__main__":
    tf.test.main()
