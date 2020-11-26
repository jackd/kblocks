import abc
import os
from typing import Callable, Optional, Sequence, Union

import gin
import tensorflow as tf
import tqdm
from absl import logging

import kblocks.data.transforms.tfrecords as tfrecords_lib
from kblocks.data.transforms.core import Transform, _get_rng, _maybe_ragged_batch
from kblocks.path import expand
from kblocks.serialize import register_serializable


def iterate_over(dataset: tf.data.Dataset, desc=None):
    for _ in tqdm.tqdm(dataset, desc=desc):
        pass


@gin.configurable(module="kb.data")
@register_serializable
class CacheFactory:
    """Implementation based on `tf.data.Dataset.cache`."""

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {}

    def cache(
        self, dataset: tf.data.Dataset, cache_dir: Union[str, tf.Tensor]
    ) -> tf.data.Dataset:
        return dataset.cache(cache_dir + "/cache")

    def exists(self, cache_dir: str):
        if tf.executing_eagerly() and not tf.io.gfile.exists(cache_dir):
            tf.io.gfile.makedirs(cache_dir)
        return tf.size(tf.io.matching_files(cache_dir + "/*")) > 0

    def remove(self, cache_dir: str):
        if tf.io.gfile.isdir(cache_dir):
            logging.info(f"Removing cache_dir {cache_dir}")
            tf.io.gfile.rmtree(cache_dir)

    def parse_func(self, spec) -> Optional[Callable]:
        del self, spec


@gin.configurable(module="kb.data")
@register_serializable
class SnapshotFactory(CacheFactory):
    """Implementation based on `tf.data.experimental.snapshot`."""

    def __init__(self, compression: str = "AUTO"):
        self._compression = compression

    def get_config(self):
        return dict(compression=self._compression)

    def cache(
        self, dataset: tf.data.Dataset, cache_dir: Union[str, tf.Tensor]
    ) -> tf.data.Dataset:
        return dataset.apply(
            tf.data.experimental.snapshot(cache_dir, compression=self._compression)
        )


@gin.configurable(module="kb.data")
@register_serializable
class SaveLoadFactory(CacheFactory):
    """Implementation based on `tf.data.experimental.[save,load]`."""

    def __init__(self, compression: Optional[str] = None):
        self._compression = compression

    def get_config(self):
        return dict(compression=self._compression)

    def cache(
        self, dataset: tf.data.Dataset, cache_dir: Union[str, tf.Tensor]
    ) -> tf.data.Dataset:
        if not self.exists(cache_dir):
            tf.print(  # pylint: disable=redundant-keyword-arg
                "Saving dataset to", cache_dir, output_stream=logging.info
            )
            tf.data.experimental.save(dataset, cache_dir, self._compression)

        return tf.data.experimental.load(
            cache_dir, dataset.element_spec, compression=self._compression,
        )


@gin.configurable(module="kb.data")
@register_serializable
class ShardedSaveLoadFactory(SaveLoadFactory):
    def __init__(self, *args, num_shards: int, shuffle: bool = False, **kwargs):
        self._num_shards = num_shards
        self._shuffle = shuffle
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(dict(num_shards=self._num_shards, shuffle=self._shuffle))

    def cache(self, dataset: tf.data.Dataset, cache_dir: Union[str, tf.Tensor]):
        dataset = dataset.enumerate()
        if not self.exists(cache_dir):
            tf.print(  # pylint: disable=redundant-keyword-arg
                "Saving dataset to", cache_dir, output_stream=logging.info
            )

            def shard_func(i, x):
                del x
                return i % self._num_shards

            tf.data.experimental.save(
                # dataset.enumerate(),
                dataset,
                cache_dir,
                compression=self._compression,
                shard_func=shard_func,
            )

        def reader_func(datasets):
            if self._shuffle:
                datasets = datasets.shuffle(self._num_shards)

            return datasets.interleave(
                # lambda x: x.map(lambda i, *el: el),
                lambda x: x,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        return tf.data.experimental.load(
            cache_dir,
            dataset.element_spec,
            compression=self._compression,
            reader_func=reader_func,
        ).map(lambda i, el: el)


@gin.configurable(module="kb.data")
@register_serializable
class TFRecordsFactory(CacheFactory):
    """Implementation based on `kblocks.data.cache.tfrecords` ops."""

    def __init__(self, compression: Optional[str] = None):
        self._compression = compression

    def get_config(self):
        return dict(compression=self._compression)

    def cache(
        self, dataset: tf.data.Dataset, cache_dir: Union[str, tf.Tensor]
    ) -> tf.data.Dataset:
        cardinality = dataset.cardinality()
        path = cache_dir + "/serialized.tfrecords"  # must work in graph mode
        if not self.exists(cache_dir):
            tf.print(  # pylint: disable=redundant-keyword-arg
                "Saving tfrecords dataset to ", cache_dir, output_stream=logging.info
            )
            tfrecords_lib.save(dataset, path, compression=self._compression)
        dataset = tfrecords_lib.load(path, compression=self._compression)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
        return dataset

    def parse_func(self, spec) -> Callable:
        return tfrecords_lib.deserializer(spec)


def get_factory(identifier) -> CacheFactory:
    if isinstance(identifier, CacheFactory):
        return identifier
    factory = tf.keras.utils.deserialize_keras_object(identifier)
    if not isinstance(factory, CacheFactory):
        raise ValueError(f"Invalid factory: {factory}")
    return factory


@gin.configurable(module="kb.data")
@register_serializable
class Cache(Transform):
    def __init__(self, cache_dir: str, factory: Optional[CacheFactory] = None):
        if factory is None:
            self._factory = CacheFactory()
        else:
            self._factory = get_factory(factory)
        self._cache_dir = expand(cache_dir)
        super().__init__()

    def get_config(self):
        return dict(
            factory=tf.keras.utils.serialize_keras_object(self._factory),
            cache_dir=self._cache_dir,
        )

    def __call__(self, dataset: tf.data.Dataset):
        parse_func = self._factory.parse_func(dataset.element_spec)
        dataset = self._factory.cache(dataset, self._cache_dir)
        if parse_func is not None:
            dataset = dataset.map(parse_func)
        return dataset


def _iter_dataset(dataset: tf.data.Dataset):
    if tf.executing_eagerly():
        yield from dataset
    else:
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    yield sess.run(it)
            except tf.errors.ResourceExhaustedError:
                pass


@gin.configurable(module="kb.data")
class RepeatedCache(Transform, abc.ABC):
    """
    A transform that represents one of multiple repeats of a cached dataset.

    This is relevant for performing data augmentation with caching. Transformed datasets
    will have the same cardinality (length) as the input dataset, but may differ over
    repeated iterations.
    """

    def __init__(
        self,
        num_repeats: int,
        path: str,
        cache_factory: CacheFactory,
        seed: Optional[int] = None,
    ):
        self._num_repeats = num_repeats
        self._seed = seed
        self._cache_factory = get_factory(cache_factory)
        self._path = expand(path)
        self._rng = None
        super().__init__()

    def get_config(self):
        return dict(
            num_repeats=self._num_repeats,
            path=self._path,
            seed=self._seed,
            cache_factory=tf.keras.utils.serialize_keras_object(self._cache_factory),
        )

    def _prepare(self) -> Sequence[str]:
        cache_dirs = [
            os.path.join(self._path, f"repeat-{i}") for i in range(self._num_repeats)
        ]
        for cache_dir in cache_dirs:
            if not tf.io.gfile.isdir(cache_dir):
                tf.io.gfile.makedirs(cache_dir)
        return cache_dirs


@gin.configurable(module="kb.data")
@register_serializable
class ChooseFromRepeatedCache(RepeatedCache):
    """
    {}

    This implementation chooses examples from each repeated cache on an
    example-by-example basis. Compared to `RandomRepeatedCache` this gives better
    shuffling but is potentially slower.
    """.format(
        RepeatedCache.__doc__
    )

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        cardinality = dataset.cardinality()
        parse_func = self._cache_factory.parse_func(dataset.element_spec)
        cache_dirs = self._prepare()

        dir_dataset = tf.data.Dataset.from_tensor_slices(cache_dirs)
        dataset = dir_dataset.interleave(
            lambda cache_dir: self._cache_factory.cache(dataset, cache_dir),
            cycle_length=self._num_repeats,
            block_length=1,
        )

        # shouldn't be any remainder
        dataset = _maybe_ragged_batch(dataset, self._num_repeats, drop_remainder=True)

        @tf.function
        def map_func(*args):
            if len(args) == 1:
                (args,) = args
            if self._rng is None:
                self._rng = _get_rng(self._seed)
            i = self._rng.uniform((), maxval=self._num_repeats, dtype=tf.int64)
            example = tf.nest.map_structure(lambda arg: tf.gather(arg, i, axis=0), args)
            if parse_func is not None:
                example = parse_func(example)
            return example

        dataset = dataset.map(map_func, num_parallel_calls=1)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
        return dataset


@gin.configurable(module="kb.data")
@register_serializable
class RandomRepeatedCache(RepeatedCache):
    """
    {}

    This implementation chooses cached epochs randomly. Compared to
    `ChooseFromRepeatedCache` this gives worse shuffling but may be faster.
    """.format(
        RepeatedCache.__doc__
    )

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        cardinality = dataset.cardinality()
        parse_func = self._cache_factory.parse_func(dataset.element_spec)
        cache_dirs = self._prepare()

        @tf.function
        def map_func(dirs: tf.Tensor) -> tf.data.Dataset:
            if self._rng is None:
                self._rng = _get_rng(self._seed)
            i = self._rng.uniform((), maxval=self._num_repeats, dtype=tf.int64)
            return self._cache_factory.cache(dataset, dirs[i])

        dataset = tf.data.Dataset.from_tensors(cache_dirs).flat_map(map_func)
        if parse_func is not None:
            dataset = dataset.map(parse_func, num_parallel_calls=1)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
        return dataset


@gin.configurable(module="kb.data")
@register_serializable
class RepeatedTFRecordsCache(Transform):
    def __init__(
        self,
        path: str,
        num_repeats: int,
        compression: Optional[str] = None,
        seed: Optional[int] = None,
        num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
        num_parallel_reads: int = tf.data.experimental.AUTOTUNE,
    ):
        self._path = expand(path)
        self._num_repeats = num_repeats
        self._compression = compression
        self._seed = seed
        self._num_parallel_calls = num_parallel_calls
        self._num_parallel_reads = num_parallel_reads
        self._rng = None
        super().__init__()

    def get_config(self):
        return dict(
            path=self._path,
            num_repeats=self._num_repeats,
            compression=self._compression,
            num_parallel_calls=self._num_parallel_calls,
            num_parallel_reads=self._num_parallel_reads,
            seed=self._seed,
        )

    def _create(self, dataset: tf.data.Dataset):
        cardinality = dataset.cardinality()
        if cardinality == tf.data.INFINITE_CARDINALITY:
            raise ValueError("Cannot cache dataset with infinite cardinality")

        if not tf.io.gfile.isdir(self._path):
            tf.io.gfile.makedirs(self._path)
        if self._num_repeats > 10000:
            raise ValueError("Only up to 9999 repeats supported")
        paths = [
            self._path + f"/serialized-{i:04d}.tfrecords"
            for i in range(self._num_repeats)
        ]
        for path in paths:
            if tf.shape(tf.io.matching_files(path))[0] == 0:
                logging.info(f"Saving tfrecords dataset to {path}")
                tfrecords_lib.save(dataset, path, compression=self._compression)
        deserializer = tfrecords_lib.deserializer(dataset.element_spec)
        return paths, cardinality, deserializer

    def __call__(self, dataset: tf.data.Dataset):
        paths, cardinality, deserializer = self._create(dataset)
        cardinality *= len(paths)

        # def shuffle(paths):
        #     if self._rng is None:
        #         self._rng = _get_rng(self._seed)
        #     i = self._rng.uniform((self._num_repeats,))
        #     perm = tf.argsort(i)
        #     return tf.gather(paths, perm, axis=0)

        return (
            # tf.data.Dataset.from_tensors(paths)
            # .map(shuffle)
            # .unbatch()  # HACK
            tf.data.Dataset.from_tensor_slices(paths)
            .flat_map(
                lambda path: tf.data.TFRecordDataset(
                    path, compression_type=self._compression
                )
            )
            .map(deserializer)
            .apply(tf.data.experimental.assert_cardinality(cardinality))
        )


@gin.configurable(module="kb.data")
@register_serializable
class ChooseFromRepeatedTFRecordsCache(RepeatedTFRecordsCache):
    """Similar to `ChooseFromRepeatedCache` but optimized for `TFRecordsFactory`."""

    def __call__(self, dataset: tf.data.Dataset):
        paths, cardinality, deserializer = self._create(dataset)
        dataset = (
            tf.data.TFRecordDataset(
                paths,
                num_parallel_reads=self._num_parallel_reads,
                compression_type=self._compression,
            )
            .batch(self._num_repeats)
            .apply(tf.data.experimental.assert_cardinality(cardinality))
        )

        def map_func(string_batch: tf.Tensor):
            if self._rng is None:
                self._rng = _get_rng(self._seed)
            i = self._rng.uniform((), maxval=self._num_repeats, dtype=tf.int64)
            string = tf.gather(string_batch, i, axis=0)
            return deserializer(string)

        return dataset.map(map_func, self._num_parallel_calls)


@gin.configurable(module="kb.data")
@register_serializable
class RandomRepeatedTFRecordsCache(RepeatedTFRecordsCache):
    def __call__(self, dataset: tf.data.Dataset):
        paths, cardinality, deserializer = self._create(dataset)

        def map_func(paths):
            if self._rng is None:
                self._rng = _get_rng(self._seed)
            i = self._rng.uniform((), maxval=self._num_repeats, dtype=tf.int64)
            return tf.data.TFRecordDataset(
                paths[i],
                num_parallel_reads=self._num_parallel_reads,
                compression_type=self._compression,
            )

        return (
            tf.data.Dataset.from_tensors(paths)
            .flat_map(map_func)
            .map(deserializer, self._num_parallel_calls)
            .apply(tf.data.experimental.assert_cardinality(cardinality))
        )
