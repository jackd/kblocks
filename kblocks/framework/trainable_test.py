import itertools
import os
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from kblocks.extras.cache import (
    BaseCacheManager,
    RepeatCacheManager,
    SnapshotManager,
    TFRecordsCacheManager,
)
from kblocks.framework.batchers import RectBatcher
from kblocks.framework.sources import BaseSource, PipelinedSource
from kblocks.framework.trainable import Trainable

# from absl.testing import parameterized


def get_model(num_features, num_targets, hidden=10, dropout=False, seed=0):
    tf.random.set_seed(seed)
    inp = tf.keras.Input((num_features,))
    x = tf.keras.layers.Dense(hidden, activation="relu")(inp)
    if dropout:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_targets)(x)
    model = tf.keras.Model(inp, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


def get_data(num_features, num_targets, train_size, val_size, seed=0):
    rng = tf.random.Generator.from_seed(seed)

    def get_data(num_examples):
        return (
            rng.uniform((num_examples, num_features), dtype=tf.float32),
            rng.uniform((num_examples, num_targets), dtype=tf.float32),
        )

    return {"train": get_data(train_size), "val": get_data(val_size)}


def get_source(
    data,
    batch_size=8,
    shuffle=False,
    random_map=False,
    use_rng=False,
    seed=0,
    shuffle_seed=0,
):
    train_ds = tf.data.Dataset.from_tensor_slices(data["train"])
    if shuffle:
        train_ds = train_ds.shuffle(len(train_ds), seed=shuffle_seed)

    modules = {}

    if random_map:
        if use_rng:
            rng = tf.random.Generator.from_seed(seed)
            modules["train_rng"] = rng

            def map_fn(x, labels):
                return x + rng.normal(tf.shape(x), dtype=tf.float32), labels

        else:
            tf.random.set_seed(seed)

            def map_fn(x, labels):
                return x + tf.random.normal(tf.shape(x), dtype=tf.float32), labels

        train_ds = train_ds.map(map_fn)

    datasets = {
        "train": train_ds.batch(batch_size),
        "validation": tf.data.Dataset.from_tensor_slices(data["val"]).batch(batch_size),
    }
    source = BaseSource(datasets.get, modules=modules)
    return source


dropout_opts = (
    False,
    # True,
)
restart_opts = (
    False,
    True,
)
shuffle_opts = (
    False,
    # True,
)
random_map_opts = (
    False,
    True,
)
use_rng_opts = (
    # False,
    True,
)
all_args = tuple(
    itertools.product(
        dropout_opts, restart_opts, shuffle_opts, random_map_opts, use_rng_opts
    )
)
all_args = (
    *all_args,
    (False, False, False, True, True),  # tf.random without restarts
    (False, False, True, False, False),  # shuffling without restarts
    (True, False, False, False, False),  # dropout without restarts
    (True, False, True, True, True),  # everything without restarts
    (True, False, True, True, True),  # everything + rng without restarts
    # (True, False, False, False, False),  # dropout
    # (True, True, False, False, False),  # dropout + restarts
)


# class TrainableTest(tf.test.TestCase, parameterized.TestCase):
class TrainableTest(tf.test.TestCase):
    def test_deterministic_training(self):
        # for args in all_args:
        #     try:
        #         self._test_deterministic_training(*args)
        #     except Exception as e:
        #         raise Exception(
        #             f"test_deterministic_training failed with args {args}"
        #         ) from e
        self._test_deterministic_cache_training()

    def _test_deterministic_cache_training(self):
        """Test different cache_manager implementations give the same results."""
        kwargs = dict(num_features=5, num_targets=1)
        model_seed = 1
        data_seed = 2
        rng_seed = 3
        num_repeats = 2
        batch_size = 16
        train_size = 100
        val_size = 20

        def get_weights(manager_impl):
            data = get_data(
                train_size=train_size, val_size=val_size, seed=data_seed, **kwargs
            )
            rng = tf.random.Generator.from_seed(rng_seed)

            def map_fn(x, y):
                return tf.cast(x, tf.float32) + rng.uniform(tf.shape(x)), y

            datasets = {
                "train": tf.data.Dataset.from_tensor_slices(data["train"]).map(map_fn),
                "validation": tf.data.Dataset.from_tensor_slices(data["val"]),
            }

            base_source = BaseSource(datasets.get, modules={"train_rng": rng})

            model = get_model(seed=model_seed, **kwargs)
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
                source = PipelinedSource(
                    base_source,
                    cache_managers=cache_managers,
                    batcher=RectBatcher(batch_size),
                    shuffle_seed=0,
                )
                trainable = Trainable(source, model)
                trainable.fit(epochs=num_repeats)
            return [w.numpy() for w in model.weights]

        def assert_all_equal(weights0, weights1):
            for w0, w1 in zip(weights0, weights1):
                np.testing.assert_allclose(w0, w1, rtol=1e-2)

        weights0 = get_weights(None)
        assert_all_equal(get_weights(None), weights0)
        assert_all_equal(get_weights(BaseCacheManager), weights0)

        # assert_all_equal(
        #     get_weights(functools.partial(SnapshotManager, preprocess=False)), weights0,
        # )
        # assert_all_equal(
        #     get_weights(functools.partial(SnapshotManager, preprocess=True)), weights0,
        # )
        assert_all_equal(get_weights(TFRecordsCacheManager), weights0)

    def _test_deterministic_training(
        self,
        dropout: bool,
        restart: bool,
        shuffle: bool,
        random_map: bool,
        use_rng: bool,
    ):
        # Setting seeds is a side-effect, so run these all in a single test.
        kwargs = dict(num_features=5, num_targets=1)
        source_seed = 0
        model_seed = 1
        data_seed = 2
        data = get_data(train_size=100, val_size=20, seed=data_seed, **kwargs)

        def get_weights(epochs, model_dir=None):
            batch_size = 8
            source = get_source(
                data,
                shuffle=shuffle,
                random_map=random_map,
                use_rng=use_rng,
                batch_size=batch_size,
                seed=source_seed,
            )
            model = get_model(seed=model_seed, dropout=dropout, **kwargs)
            initial_weights = [w.numpy() for w in model.weights]
            trainable = Trainable(source, model, model_dir)
            trainable.fit(epochs=epochs)
            return initial_weights, model.weights

        def assert_all_equal(x, y):
            assert len(x) == len(y)
            for xi, yi in zip(x, y):
                np.testing.assert_equal(xi, yi)

        w00, weights0 = get_weights(2)
        if restart:
            with TemporaryDirectory() as model_dir:
                get_weights(1, model_dir)
                w10, weights1 = get_weights(2, model_dir)
                assert_all_equal(w00, w10)
        else:
            w11, weights1 = get_weights(2)
            assert_all_equal(w00, w11)
        weights0, weights1 = self.evaluate((weights0, weights1))
        assert_all_equal(weights0, weights1)


if __name__ == "__main__":
    # unittest.main()
    tf.test.main()
    # TrainableTest().test_deterministic_training(True, False, True, True)
