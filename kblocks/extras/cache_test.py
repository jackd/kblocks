import functools
import os
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from absl import logging

from kblocks.extras import cache


def as_numpy(dataset: tf.data.Dataset):
    return np.array([el.numpy() for el in dataset])


class CacheManagerTest(tf.test.TestCase):
    def test_cache_managers(self):
        def get_values(manager, del_rng=False, **kwargs):
            seed = 0
            rng = tf.random.Generator.from_seed(seed)
            dataset = tf.data.Dataset.range(100).map(
                lambda x: tf.cast(x, tf.float32) + rng.uniform(())
            )
            with TemporaryDirectory() as tmp_dir:
                man = (
                    None
                    if manager is None
                    else manager(os.path.join(tmp_dir, "cache"), **kwargs)
                )
                ds = dataset if man is None else man(dataset)
                if del_rng:
                    del rng
                values = as_numpy(ds)
            return values

        v0 = get_values(None)

        def test_manager(manager, **kwargs):
            values = get_values(manager, **kwargs)
            np.testing.assert_equal(v0, values)

        test_manager(None)
        test_manager(cache.BaseCacheManager, preprocess=False)
        test_manager(cache.BaseCacheManager, preprocess=True)
        test_manager(cache.SaveLoadManager)
        test_manager(cache.TFRecordsCacheManager)
        test_manager(cache.SnapshotManager, preprocess=False)
        test_manager(cache.SnapshotManager, preprocess=True)

        test_manager(cache.BaseCacheManager, preprocess=True, del_rng=True)
        test_manager(cache.SaveLoadManager, del_rng=True)
        test_manager(cache.TFRecordsCacheManager, del_rng=True)
        test_manager(cache.SnapshotManager, preprocess=True, del_rng=True)

    def test_repeat_managers(self):
        seed = 0
        num_repeats = 2
        base_length = 100

        def get_values(manager, del_rng=False, **kwargs):
            rng = tf.random.Generator.from_seed(seed)
            dataset = tf.data.Dataset.range(base_length).map(
                lambda x: tf.cast(x, tf.float32) + rng.uniform(())
            )
            with TemporaryDirectory() as tmp_dir:
                if manager is None:
                    ds = dataset.repeat(num_repeats)
                else:
                    man = cache.RepeatCacheManager(
                        tmp_dir,
                        num_repeats,
                        functools.partial(manager, **kwargs),
                        take_single=False,
                        shuffle_datasets=False,
                    )
                    ds = man(dataset)
                if del_rng:
                    del rng
                values = as_numpy(ds).reshape((num_repeats, base_length))
            return values

        expected = get_values(None)
        # ensure data augmentation is different across epochs

        for i in range(1, num_repeats):
            assert not np.all(expected[0] == expected[i])

        def test_manager(manager, **kwargs):
            values = get_values(manager, **kwargs)
            np.testing.assert_equal(expected, values)

        test_manager(None)
        test_manager(cache.BaseCacheManager, preprocess=False)
        test_manager(cache.BaseCacheManager, preprocess=True)
        test_manager(cache.SaveLoadManager)
        test_manager(cache.TFRecordsCacheManager)
        test_manager(cache.SnapshotManager, preprocess=False)

        # preprocessing is offline, so rng should be able to be deleted safely
        test_manager(cache.BaseCacheManager, preprocess=True, del_rng=True)
        test_manager(cache.TFRecordsCacheManager, del_rng=True)
        test_manager(cache.SaveLoadManager, del_rng=True)

        if tf.version.VERSION.startswith("2.4.0-dev"):
            logging.warning(
                "Skipping SnapshotManager(preprocess=True) test - "
                "see https://github.com/tensorflow/tensorflow/issues/44278"
            )
        else:
            test_manager(cache.SnapshotManager, preprocess=True)
            test_manager(cache.SnapshotManager, preprocess=True, del_rng=True)


if __name__ == "__main__":
    tf.test.main()
