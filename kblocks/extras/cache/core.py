import abc
import os
from typing import Callable

import gin
import tensorflow as tf
from absl import logging
from tqdm import tqdm

from kblocks.utils import identity

# import wtftf

AUTOTUNE = tf.data.experimental.AUTOTUNE
Transform = Callable[[tf.data.Dataset], tf.data.Dataset]


def graph_dataset_iterator(dataset):
    example_tf = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session() as sess:
        try:
            while True:
                yield sess.run(example_tf)
        except tf.errors.OutOfRangeError:
            pass


def dataset_iterator(dataset):
    if tf.executing_eagerly():
        return dataset
    return graph_dataset_iterator(dataset)


def cache_exists(path: str):
    return os.path.isfile(f"{path}.index")


def iterate_over(dataset, desc=None):
    for _ in tqdm(dataset_iterator(dataset), desc=desc):
        pass


def preprocessed_cache(dataset, path, desc=None):
    dataset = dataset.cache(path)
    if cache_exists(path):
        logging.info(f"Reusing cache data at {path}")
    else:
        if desc is None:
            desc = f"Creating cache data at {path}"
        iterate_over(dataset, desc=desc)
    return dataset


class CacheManager(abc.ABC):
    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def __call__(
        self, dataset: tf.data.Dataset, transform: Transform = identity
    ) -> tf.data.Dataset:
        raise NotImplementedError("Abstract method")


@gin.configurable(module="kb.cache")
class BaseCacheManager(CacheManager):
    def __init__(self, cache_dir: str, preprocess: bool = False):
        self._cache_dir = cache_dir
        self._preprocess = preprocess
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        elif not os.path.isdir(cache_dir):
            raise ValueError(f"non-directory exists at {cache_dir}")

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    def clear(self):
        """Clear the cache."""
        if os.path.exists(self.cache_dir):
            tf.io.gfile.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)

    def __call__(
        self, dataset: tf.data.Dataset, transform: Transform = identity
    ) -> tf.data.Dataset:
        path = os.path.join(self.cache_dir, "cache")
        if self._preprocess:
            dataset = preprocessed_cache(dataset, path)
        else:
            dataset = dataset.cache(path)
        return transform(dataset)


@gin.configurable(module="kb.cache")
class SnapshotManager(CacheManager):
    def __init__(self, path: str, compression: str = "AUTO", preprocess: bool = False):
        if preprocess and tf.version.VERSION.startswith("2.4.0-dev"):
            logging.warning(
                "SnapshotManager with preprocess non-deterministic in tf-nightly - see "
                "https://github.com/tensorflow/tensorflow/issues/44278"
            )
        self._path = path
        self._compression = compression
        self._preprocess = preprocess

    @property
    def path(self) -> str:
        return self._path

    @property
    def compression(self) -> str:
        return self._compression

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    def clear(self):
        """Clear the cache."""
        if os.path.exists(self.path):
            tf.io.gfile.remove(self.path)

    def __call__(
        self, dataset: tf.data.Dataset, transform: Transform = identity
    ) -> tf.data.Dataset:
        dataset = dataset.apply(
            tf.data.experimental.snapshot(self.path, compression=self.compression)
        )

        if self.preprocess and not os.path.isdir(self.path):
            assert tf.executing_eagerly()
            # iterate over dataset to ensure data saved to disk
            for _ in tqdm(dataset, desc=f"Preprocessing snapshot at {self.path}"):
                pass
        return transform(dataset)


@gin.configurable(module="kb.cache")
class SaveLoadManager(CacheManager):
    """CacheManager implementation that uses `tf.data.experimental.[save|load]`."""

    def __init__(self, path: str, compression: str = "GZIP"):
        self._path = path
        self._compression = compression

    @property
    def path(self) -> str:
        return self._path

    @property
    def compression(self) -> str:
        return self._compression

    def clear(self):
        """Clear the cache."""
        if os.path.exists(self.path):
            tf.io.gfile.rmtree(self.path)

    def __call__(self, dataset: tf.data.Dataset, transform: Transform = identity):
        if not os.path.exists(self.path):
            try:
                logging.info(f"Saving dataset to {self.path}")
                tf.data.experimental.save(dataset, self.path, self.compression)
            except (Exception, KeyboardInterrupt):
                if os.path.exists(self.path):
                    self.clear()
                raise
        dataset = tf.data.experimental.load(
            self.path, dataset.element_spec, compression=self.compression
        )
        return transform(dataset)


@gin.configurable(module="kb.cache")
class RepeatCacheManager(CacheManager):
    """
    Manager for multiple repeats of a dataset e.g. for caching after data augmentation.

    In general, using `take_single=True` is slightly more efficient as it only applies
    custom `transform` (which may contain e.g. shuffling) to the taken dataset, rather
    than each of the constituents. However, this requires `shuffle_datasets` to be True,
    which will give epochs out-of-order compared to `dataset.repeat(num_repeats)`.

    You may also consider using `take_single=False, shuffle_files=True` and apply
    transform manually on the returned dataset. This gives a dataset with `num_repeats`
    times as many elements as the original (assuming no batching occurs in the
    subsequent transform), but makes it impossible to have the same batching dynamics
    at the end of each epoch, e.g. with `num_repeats=2`, initial dataset length of 10
    a subsequent `batch(4)`, the returned dataset will have 5 elements of batch size
    [4, 4, 4, 4, 4], as opposed to manually repeating after the batch which would give
    [4, 4, 2, 4, 4, 2].

    Args:
        cache_dir: root directory to save each epoch.
        num_repeats: number of repeats to save.
        manager_impl: function producing a CacheManager from a save directory.
        take_single: if True, the resulting dataset is the same length as the initial
            dataset, otherwise it will be `num_repeats` times as long.
        shuffle_datasets: if True, the dataset list will be shuffled (as opposed to the
            constituent datasets themselves).
    """

    def __init__(
        self,
        cache_dir: str,
        num_repeats: int,
        manager_impl: Callable[[str], CacheManager] = BaseCacheManager,
        take_single: bool = True,
        shuffle_datasets: bool = True,
    ):
        self._cache_dir = cache_dir
        self._managers = tuple(
            manager_impl(os.path.join(cache_dir, f"repeat-{i:04d}-{num_repeats:04d}"),)
            for i in range(num_repeats)
        )
        if take_single and not shuffle_datasets:
            raise ValueError("`shuffle_datasets` must be True if `take_singe` is")
        self._take_single = take_single
        self._shuffle_datasets = shuffle_datasets

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    def clear(self):
        for manager in self._managers:
            manager.clear()

    def __call__(
        self, dataset: tf.data.Dataset, transform: Transform = identity
    ) -> tf.data.Dataset:
        # datasets = [manager(dataset) for manager in self._managers]
        # length = len(dataset)
        # choices = tf.data.Dataset.range(length).map(
        #     lambda x: wtftf.random.uniform(
        #         (), maxval=len(self._managers), dtype=tf.int64
        #     )
        # )
        # return tf.data.experimental.choose_from_datasets(datasets, choices)

        trans0, trans1 = (
            (identity, transform) if self._take_single else (transform, identity)
        )
        datasets = [manager(dataset, trans0) for manager in self._managers]
        dataset = tf.data.Dataset.from_tensor_slices(datasets)
        if self._shuffle_datasets:
            dataset = dataset.shuffle(len(datasets))
        if self._take_single:
            dataset = dataset.take(1)
        dataset = dataset.flat_map(identity)
        dataset = trans1(dataset)
        return dataset


@gin.configurable(module="kb.cache")
def cache_managers(
    root_dir, train_impl=BaseCacheManager, validation_impl=BaseCacheManager
):
    out = {}
    if train_impl is not None:
        out["train"] = train_impl(os.path.join(root_dir, "train"))
    if validation_impl is not None:
        out["validation"] = validation_impl(os.path.join(root_dir, "validation"))
    return out
