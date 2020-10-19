import abc
import os
from typing import Iterable

import gin
import tensorflow as tf
from absl import logging
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE


def graph_dataset_iterator(dataset):
    example_tf = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session() as sess:
        try:
            while True:
                yield sess.run(example_tf)
        except tf.errors.OutOfRangeError:
            pass


def dataset_iterator(dataset, as_numpy=None):
    if hasattr(dataset, "as_numpy_iterator"):
        return dataset.as_numpy_iterator()
    if tf.executing_eagerly():
        if as_numpy:
            return (tf.nest.map_structure(lambda x: x.numpy(), d) for d in dataset)
        else:
            return dataset
    else:
        if as_numpy is False:
            raise ValueError("Cannot get non-numpy iterator in graph mode")
        return graph_dataset_iterator(dataset)


def cache_exists(path: str):
    return os.path.isfile(f"{path}.index")


def iterate_over(dataset, desc=None):
    for _ in tqdm(dataset_iterator(dataset), desc=desc):
        pass


def preprocessed_cache(dataset, path, desc="Creating cache..."):
    dataset = dataset.cache(path)
    if cache_exists(path):
        logging.info(f"Reusing cache data at {path}")
    else:
        iterate_over(dataset, desc=desc)
    return dataset


class CacheManager(abc.ABC):
    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
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
        tf.io.gfile.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)

    def __call__(self, dataset: tf.data.Dataset):
        path = os.path.join(self.cache_dir, "cache")
        if self._preprocess:
            return preprocessed_cache(dataset, path)
        return dataset.cache(path)


@gin.configurable(module="kb.cache")
class SnapshotManager(CacheManager):
    def __init__(self, path: str, compression: str = "AUTO", preprocess: bool = False):
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
        tf.io.gfile.remove(self._path)

    def __call__(self, dataset: tf.data.Dataset):
        dataset = dataset.apply(
            tf.data.experimental.snapshot(self.path, compression=self.compression)
        )
        if self.preprocess and not os.path.isdir(self.path):
            assert tf.executing_eagerly()
            # iterate over dataset to make data saved to disk
            for _ in tqdm(dataset, desc=f"Preprocessing snapshot at {self.path}"):
                pass
        return dataset


@gin.configurable(module="kb.cache")
class SaveLoadManager(CacheManager):
    """CacheManager implementation that uses `tf.data.experimental.save/load`."""

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
        tf.io.gfile.remove(self.path)

    def __call__(self, dataset: tf.data.Dataset):
        if not os.path.exists(self.path):
            try:
                logging.info(f"Saving dataset to {self.path}")
                tf.data.experimental.save(dataset, self.path, self.compression)
            except (Exception, KeyboardInterrupt):
                if os.path.exists(self.path):
                    self.clear()
                raise
        return tf.data.experimental.load(
            self.path, dataset.element_spec, compression=self.compression
        )


def _identity(x):
    return x


@gin.configurable(module="kb.cache")
class ParentManager(CacheManager):
    def __init__(
        self, cache_dir: str, num_children: int, manager_impl=BaseCacheManager,
    ):
        self._cache_dir = cache_dir
        self._managers = tuple(
            manager_impl(os.path.join(cache_dir, f"repeat-{i:04d}-{num_children:04d}"),)
            for i in range(num_children)
        )

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    def clear(self):
        for manager in self._managers:
            manager.clear()

    @abc.abstractmethod
    def _children(self, dataset: tf.data.Dataset) -> Iterable[tf.data.Dataset]:
        raise NotImplementedError()

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        datasets = [
            manager(dataset)
            for (manager, dataset) in zip(self._managers, self._children(dataset))
        ]
        return tf.data.Dataset.from_tensor_slices(datasets).flat_map(_identity)


@gin.configurable(module="kb.cache")
class ShardedCacheManager(ParentManager):
    def __init__(self, cache_dir: str, num_shards: int, **kwargs):
        self._num_shards = num_shards
        super().__init__(cache_dir=cache_dir, num_children=num_shards, **kwargs)

    def _children(self, dataset):
        return [dataset.shard(self._num_shards, i) for i in range(self._num_shards)]


@gin.configurable(module="kb.cache")
class RepeatCacheManager(ParentManager):
    def __init__(self, cache_dir: str, num_repeats: int, **kwargs):
        self._num_repeats = num_repeats
        super().__init__(cache_dir=cache_dir, num_children=num_repeats, **kwargs)

    def _children(self, dataset):
        return [dataset] * self._num_repeats


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
