import functools
import os
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from absl import logging

from kblocks.extras.cache import (
    BaseCacheManager,
    RepeatCacheManager,
    SaveLoadManager,
    SnapshotManager,
    TFRecordsCacheManager,
)
from kblocks.framework.batchers import RectBatcher
from kblocks.framework.sources import BaseSource, PipelinedSource


def map_fn(x, y, rng=None, std=1e-3):
    x = tf.cast(x, tf.float32)
    if rng is not None:
        x = x + rng.normal(tf.shape(x), stddev=std)
    return x, y


def get_data(num_features, num_targets, train_size, val_size, seed=0):
    del seed
    # rng = tf.random.Generator.from_seed(seed)

    def get_split_data(num_examples):
        return (
            tf.reshape(
                tf.range(num_examples * num_features), (num_examples, num_features)
            ),
            tf.reshape(
                tf.range(num_examples * num_targets), (num_examples, num_targets)
            ),
            # rng.uniform((num_examples, num_features), dtype=tf.float32),
            # rng.uniform((num_examples, num_targets), dtype=tf.float32),
        )

    return {"train": get_split_data(train_size), "validation": get_split_data(val_size)}


class PipelinedTest(tf.test.TestCase):
    def test_deterministic(self):
        num_features = 5
        num_targets = 1
        train_size = 100
        val_size = 20
        batch_size = 16
        num_repeats = 2

        rng_seed = 0

        rng = tf.random.Generator.from_seed(seed=rng_seed)

        data = get_data(num_features, num_targets, train_size, val_size)
        datasets = {k: tf.data.Dataset.from_tensor_slices(v) for k, v in data.items()}
        datasets["train"] = datasets["train"].map(functools.partial(map_fn, rng=rng))
        datasets["validation"] = datasets["validation"].map(map_fn)

        def get_source_data(manager_impl):
            tf.random.set_seed(0)
            rng.reset_from_seed(rng_seed)
            with TemporaryDirectory() as tmp_dir:
                if manager_impl is None:
                    cache_managers = None
                else:
                    cache_managers = {
                        "train": RepeatCacheManager(
                            os.path.join(tmp_dir, "train"),
                            num_repeats=num_repeats,
                            manager_impl=manager_impl,
                            take_single=False,
                            shuffle_datasets=False,
                        ),
                        "validation": None,
                    }
                base_source = BaseSource(datasets.get, modules=dict(rng=rng))
                source = PipelinedSource(
                    base_source,
                    cache_managers=cache_managers,
                    batcher=RectBatcher(batch_size),
                    shuffle_seed=0,
                )
                split = "train"
                dataset = source.get_dataset(split)
                length = source.epoch_length(split)
                return tuple(
                    tuple(
                        (x.numpy(), y.numpy())
                        for x, y in dataset.repeat().take(num_repeats * length)
                    )
                    for _ in range(2)
                )

        def assert_all_equal(struct0, struct1):
            tf.nest.assert_same_structure(struct0, struct1)
            tf.nest.map_structure(np.testing.assert_equal, struct0, struct1)

        data0, _ = get_source_data(None)

        def assert_data_same(manager_impl, duplicates=True, same_as_uncached=True):
            data1, data2 = get_source_data(manager_impl)
            if duplicates:
                assert_all_equal(data1, data2)  # same for each repeated iteration
            if same_as_uncached:
                assert_all_equal(data0, data1)  # same as uncached version

        assert_data_same(None, duplicates=False)
        assert_data_same(SaveLoadManager)
        assert_data_same(functools.partial(SnapshotManager, preprocess=False))
        assert_data_same(functools.partial(BaseCacheManager, preprocess=False))
        assert_data_same(TFRecordsCacheManager)
        assert_data_same(functools.partial(BaseCacheManager, preprocess=True))
        if tf.version.VERSION.startswith("2.4.0-dev"):
            logging.warning(
                "Skipping SnapshotManager(preprocess=True) test - "
                "see https://github.com/tensorflow/tensorflow/issues/44278"
            )
        else:
            assert_data_same(functools.partial(SnapshotManager, preprocess=True))


if __name__ == "__main__":
    tf.test.main()
