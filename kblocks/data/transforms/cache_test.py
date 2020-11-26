import itertools
import os
import tempfile
from typing import Callable

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from kblocks.data import transforms

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def as_array(dataset: tf.data.Dataset) -> np.ndarray:
    return np.array([el.numpy() for el in dataset])


lazy_factories = (
    transforms.CacheFactory(),
    transforms.SnapshotFactory(),
)

eager_factories = (
    transforms.TFRecordsFactory(),
    transforms.SaveLoadFactory(),
)

repeated_impls = (
    transforms.ChooseFromRepeatedCache,
    transforms.RandomRepeatedCache,
)

factories = lazy_factories + eager_factories


class CacheTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*factories)
    def test_cache_transform(self, factory: transforms.CacheFactory):
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
            cached = dataset.apply(transforms.Cache(tmp_dir, factory))
            np.testing.assert_equal(cached.cardinality().numpy(), epoch_length)
            for _ in range(2):
                np.testing.assert_equal(as_array(cached), expected)
                np.testing.assert_equal(rng.state.numpy(), state)

    def test_repreated_tfrecords_transform(self):
        seed = 0
        epoch_length = 5
        rng = tf.random.Generator.from_seed(seed)
        dataset = tf.data.Dataset.range(epoch_length).map(
            lambda x: tf.cast(x, tf.float32) + rng.uniform(())
        )

        num_repeats = 3

        def unique(dataset, repeats):
            return np.unique(
                np.array([as_array(dataset) for _ in range(repeats)]).flatten()
            )

        expected_unique = unique(dataset, num_repeats)
        state = rng.state.numpy()

        rng.reset_from_seed(seed)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cached = transforms.RepeatedTFRecordsCache(
                num_repeats=num_repeats, path=os.path.join(tmp_dir, "cache"),
            )(dataset)
            actual_unique = unique(cached, repeats=1)
            np.testing.assert_equal(actual_unique.size, expected_unique.size)
            np.testing.assert_equal(expected_unique, actual_unique)
            np.testing.assert_equal(rng.state.numpy(), state)

            actual_unique2 = unique(cached, repeats=2)
            np.testing.assert_equal(actual_unique2.size, actual_unique.size)
            np.testing.assert_equal(actual_unique2, actual_unique)
            np.testing.assert_equal(rng.state.numpy(), state)

    def test_choose_from_tfrecords_transform(self):
        seed = 0
        epoch_length = 5
        rng = tf.random.Generator.from_seed(seed)
        dataset = tf.data.Dataset.range(epoch_length).map(
            lambda x: tf.cast(x, tf.float32) + rng.uniform(())
        )

        num_repeats = 3

        def unique(dataset, repeats):
            return np.unique(
                np.array([as_array(dataset) for _ in range(repeats)]).flatten()
            )

        expected_unique = unique(dataset, num_repeats)
        state = rng.state.numpy()

        rng.reset_from_seed(seed)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cached = transforms.ChooseFromRepeatedTFRecordsCache(
                num_repeats=num_repeats, path=os.path.join(tmp_dir, "cache"),
            )(dataset)
            actual_unique = unique(cached, repeats=20)
            np.testing.assert_equal(actual_unique.size, expected_unique.size)
            np.testing.assert_equal(expected_unique, actual_unique)
            np.testing.assert_equal(rng.state.numpy(), state)

            actual_unique2 = unique(cached, repeats=20)
            np.testing.assert_equal(actual_unique2.size, actual_unique.size)
            np.testing.assert_equal(actual_unique2, actual_unique)
            np.testing.assert_equal(rng.state.numpy(), state)

    @parameterized.parameters(
        *itertools.product(eager_factories, repeated_impls),
        *itertools.product(lazy_factories, (transforms.RandomRepeatedCache,)),
        # lazy factories + ChooseFromRepeatedCache will not give consistent results
        # since rng is called in interleaved order
    )
    def test_repeated_cache_transform(
        self, factory: transforms.CacheFactory, repeated_impl: Callable,
    ):
        seed = 0
        epoch_length = 5
        rng = tf.random.Generator.from_seed(seed)
        dataset = tf.data.Dataset.range(epoch_length).map(
            lambda x: tf.cast(x, tf.float32) + rng.uniform(())
        )

        num_repeats = 3

        def unique(dataset, repeats):
            return np.unique(
                np.array([as_array(dataset) for _ in range(repeats)]).flatten()
            )

        expected_unique = unique(dataset, num_repeats)
        state = rng.state.numpy()

        rng.reset_from_seed(seed)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cached = repeated_impl(
                num_repeats=num_repeats,
                path=os.path.join(tmp_dir, "cache"),
                cache_factory=factory,
            )(dataset)
            actual_unique = unique(cached, repeats=20)
            np.testing.assert_equal(actual_unique.size, expected_unique.size)
            np.testing.assert_equal(expected_unique, actual_unique)
            np.testing.assert_equal(rng.state.numpy(), state)

            actual_unique2 = unique(cached, repeats=20)
            np.testing.assert_equal(actual_unique2.size, actual_unique.size)
            np.testing.assert_equal(actual_unique2, actual_unique)
            np.testing.assert_equal(rng.state.numpy(), state)


if __name__ == "__main__":
    tf.test.main()
