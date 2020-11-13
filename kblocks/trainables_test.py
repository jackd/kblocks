import itertools
import os
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import tensorflow as tf
from absl import logging

from kblocks.data import transforms
from kblocks.data.sources import DataSource
from kblocks.extras.layers.dropout import Dropout as DropoutRng
from kblocks.trainables import Trainable, fit

os.environ["TF_DETERMINISTIC_OPS"] = "1"


class InterruptCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch: int):
        self._epoch = epoch
        super().__init__()

    def on_epoch_begin(self, epoch: int, logs=None):
        if epoch == self._epoch:
            logging.info(f"Interrupting at start of epoch {epoch}")
            raise KeyboardInterrupt

    def get_config(self):
        return dict(epoch=self._epoch)

    @classmethod
    def from_config(cls, **config):
        return cls(**config)


class RandomSource(DataSource):
    def __init__(self, num_features: int, num_targets: int, size: int, seed: int):
        self.num_features = num_features
        self.num_targets = num_targets
        self.size = size
        self.seed = seed

        rng = tf.random.Generator.from_seed(seed)
        self._dataset = tf.data.Dataset.from_tensor_slices(
            (
                rng.uniform((size, num_features), dtype=tf.float32),
                rng.uniform((size, num_targets), dtype=tf.float32),
            )
        )
        super().__init__()

    @property
    def dataset(self):
        return self._dataset

    def get_config(self):
        return dict(
            num_features=self.num_features,
            num_targets=self.num_targets,
            size=self.size,
            seed=self.seed,
        )


def get_model(
    num_features, num_targets, hidden=10, dropout=False, seed=0, use_rng=False
):
    tf.random.set_seed(seed)
    with transforms.global_generator_context(tf.random.Generator.from_seed(seed)):
        inp = tf.keras.Input((num_features,))
        x = tf.keras.layers.Dense(hidden, activation="relu")(inp)
        if dropout:
            if use_rng:
                x = DropoutRng(0.5)(x)
            else:
                x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(num_targets)(x)
        model = tf.keras.Model(inp, x)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


def get_pipelined_source(
    source: DataSource,
    batch_size=8,
    shuffle=False,
    random_map=False,
    use_rng=False,
    map_seed=0,
    shuffle_seed=0,
    cache_factory: Optional[transforms.CacheFactory] = None,
    preprocess_offline: bool = True,
    cache_dir: str = "",
    num_repeats: Optional[int] = None,
):
    assert cache_factory is None or isinstance(cache_factory, transforms.CacheFactory)
    if shuffle:
        if use_rng:
            source = source.shuffle_rng(len(source.dataset), seed=shuffle_seed)

        else:
            source = source.shuffle(len(source.dataset), seed=shuffle_seed)

    if random_map:

        if use_rng:

            def map_fn(x, labels):
                return (
                    x
                    + tf.random.get_global_generator().normal(
                        tf.shape(x), dtype=tf.float32
                    ),
                    labels,
                )

            source = source.map_rng(map_fn, seed=map_seed)

        else:

            def map_fn(x, labels):
                return x + tf.random.normal(tf.shape(x), dtype=tf.float32), labels

            source = source.map(map_fn)

    if cache_factory is not None:
        if num_repeats is None:
            source = source.apply(transforms.Cache(cache_dir, cache_factory))
        else:
            source = source.apply(
                # transforms.ChooseFromRepeatedCache(
                transforms.RandomRepeatedCache(
                    num_repeats,
                    cache_dir,
                    cache_factory,
                    seed=0,
                    preprocess_offline=preprocess_offline,
                )
            )
    source = source.batch(batch_size)

    return source


dropout_opts = (
    False,
    # True,  # not sure why these don't work
)
restart_opts = (
    True,
    False,
)
shuffle_opts = (
    False,
    True,
)
random_map_opts = (
    False,
    True,
)
use_rng_opts = (
    # False,
    True,
)
rng_args = tuple(
    itertools.product((False,), restart_opts, shuffle_opts, random_map_opts, (True,))
)
dropout_args = tuple(
    itertools.product((True,), (False,), shuffle_opts, random_map_opts, (True,))
)
all_args = rng_args + dropout_args


# class TrainableTest(tf.test.TestCase, parameterized.TestCase):
class TrainableTest(tf.test.TestCase):
    def test_deterministic_training(self):
        for args in all_args:
            try:
                self._test_deterministic_training(*args)
            except Exception as e:
                raise Exception(
                    f"test_deterministic_training failed with args {args}"
                ) from e
        self._test_deterministic_cache_training()

    def _test_deterministic_cache_training(self):
        """Test different cache_manager implementations give the same results."""
        kwargs = dict(num_features=5, num_targets=1)
        model_seed = 1
        data_seed = 2
        map_seed = 3
        shuffle_seed = 4
        num_repeats = 2
        batch_size = 16
        train_size = 100
        # val_size = 20
        train_source = RandomSource(size=train_size, seed=data_seed, **kwargs)

        def get_weights(cache_factory, callbacks=(), model_dir=None):

            model = get_model(seed=model_seed, **kwargs)
            with TemporaryDirectory() as tmp_dir:
                source = get_pipelined_source(
                    train_source,
                    cache_factory=cache_factory,
                    cache_dir=tmp_dir,
                    batch_size=batch_size,
                    shuffle_seed=shuffle_seed,
                    map_seed=map_seed,
                    num_repeats=num_repeats,
                )
                trainable = Trainable(model, source, callbacks=callbacks)
                fit_callbacks = []
                if model_dir is not None:
                    fit_callbacks.append(
                        tf.keras.callbacks.experimental.BackupAndRestore(model_dir)
                    )
                fit(trainable, epochs=num_repeats)
            return [w.numpy() for w in model.weights]

        def assert_all_equal(weights0, weights1):
            for w0, w1 in zip(weights0, weights1):
                np.testing.assert_allclose(w0, w1, rtol=1e-2)

        weights0 = get_weights(None)
        assert_all_equal(get_weights(None), weights0)
        assert_all_equal(get_weights(transforms.CacheFactory()), weights0)

        assert_all_equal(
            get_weights(transforms.SnapshotFactory()), weights0,
        )
        # assert_all_equal(
        #     get_weights(functools.partial(transforms.Snapshot, preprocess=True)),
        #     weights0,
        # )
        if tf.version.VERSION > "2.4":
            # not sure why these guys don't work in 2.3...
            assert_all_equal(get_weights(transforms.TFRecordsFactory()), weights0)
            assert_all_equal(get_weights(transforms.SaveLoadFactory()), weights0)
        # Ensure pre-emptible
        with TemporaryDirectory() as model_dir:
            try:
                get_weights(
                    transforms.CacheFactory(),
                    model_dir=model_dir,
                    callbacks=[InterruptCallback(1)],
                )
            except KeyboardInterrupt:
                pass
            assert_all_equal(
                get_weights(transforms.CacheFactory(), model_dir=model_dir), weights0
            )

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
        model_seed = 1
        map_seed = 0
        shuffle_seed = 0
        train_source = RandomSource(size=100, seed=0, **kwargs)

        def get_weights(epochs, model_dir=None, callbacks=()):
            batch_size = 8
            source = get_pipelined_source(
                train_source,
                shuffle=shuffle,
                random_map=random_map,
                use_rng=use_rng,
                batch_size=batch_size,
                map_seed=map_seed,
                shuffle_seed=shuffle_seed,
            )
            model = get_model(
                seed=model_seed, dropout=dropout, use_rng=use_rng, **kwargs
            )
            initial_weights = [w.numpy() for w in model.weights]
            callbacks = list(callbacks)
            fit_callbacks = []
            if model_dir is not None:
                fit_callbacks.append(
                    tf.keras.callbacks.experimental.BackupAndRestore(model_dir),
                )

            trainable = Trainable(model, source, callbacks=callbacks)
            fit(trainable, epochs=epochs, callbacks=fit_callbacks)
            return initial_weights, model.weights

        def assert_all_equal(x, y):
            assert len(x) == len(y)
            for xi, yi in zip(x, y):
                np.testing.assert_equal(xi, yi)

        epochs = 4
        w00, weights0 = get_weights(epochs)
        if restart:
            with TemporaryDirectory() as model_dir:
                try:
                    get_weights(epochs, model_dir, [InterruptCallback(epochs // 2)])
                except KeyboardInterrupt:
                    pass
                w10, weights1 = get_weights(epochs, model_dir)
                assert_all_equal(w00, w10)
        else:
            w11, weights1 = get_weights(epochs)
            assert_all_equal(w00, w11)
        weights0, weights1 = self.evaluate((weights0, weights1))
        assert_all_equal(weights0, weights1)


if __name__ == "__main__":
    # unittest.main()
    tf.test.main()
    # TrainableTest().test_deterministic_training(True, False, True, True)
