import abc
from absl import logging
from typing import Callable, Optional, Mapping
import tensorflow as tf
import gin
from kblocks.framework.cache.core import CacheManager

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _always_throws():
    raise NotImplementedError("Should not be called")


class DataPipeline(abc.ABC):
    def processed_spec(self, example_spec):
        dataset = tf.data.Dataset.from_generator(
            _always_throws,
            output_types=tf.nest.map_structure(lambda x: x.dtype, example_spec),
            output_shapes=tf.nest.map_structure(lambda x: x.shape, example_spec),
        )
        return self(dataset, "train").element_spec

    @abc.abstractproperty
    def batch_size(self) -> int:
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def __call__(self, dataset: tf.data.Dataset, split: str) -> tf.data.Dataset:
        raise NotImplementedError("Abstract method")


@gin.configurable(module="kb.framework")
class BasePipeline(DataPipeline):
    def __init__(
        self,
        batch_size: int,
        pre_cache_map: Optional[Callable] = None,
        pre_batch_map: Optional[Callable] = None,
        post_batch_map: Optional[Callable] = None,
        cache_managers: Optional[Mapping[str, CacheManager]] = None,
        shuffle_buffer: Optional[int] = None,
        repeats: Optional[int] = None,
        prefetch_buffer: int = AUTOTUNE,
        num_parallel_calls: int = AUTOTUNE,
        clear_cache: bool = False,
        drop_remainder: bool = True,
    ):
        self._pre_cache_map = pre_cache_map
        self._pre_batch_map = pre_batch_map
        self._post_batch_map = post_batch_map

        self._batch_size = batch_size
        self._shuffle_buffer = shuffle_buffer
        self._repeats = repeats
        self._prefetch_buffer = prefetch_buffer
        self._num_parallel_calls = num_parallel_calls
        self._clear_cache = clear_cache
        self._cache_managers = cache_managers
        self._drop_remainder = drop_remainder

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def processed_spec(self, example_spec):
        dataset = tf.data.Dataset.from_generator(
            _always_throws,
            output_types=tf.nest.map_structure(lambda x: x.dtype, example_spec),
            output_shapes=tf.nest.map_structure(lambda x: x.shape, example_spec),
        )
        for m in self._pre_cache_map, self._pre_batch_map:
            if m is not None:
                dataset = dataset.map(m)
        dataset = dataset.batch(self.batch_size)
        if self._post_batch_map is not None:
            dataset = dataset.map(self._post_batch_map)
        return dataset.element_spec

    def __call__(self, dataset: tf.data.Dataset, split: str):
        return self._process_dataset(dataset, split)

    def _process_dataset(self, dataset: tf.data.Dataset, split: str) -> tf.data.Dataset:
        with tf.keras.backend.learning_phase_scope(split == "train"):
            if self._pre_cache_map is not None:
                cache_managers = (
                    None
                    if self._cache_managers is None
                    else self._cache_managers.get(split)
                )
                use_cache = cache_managers is not None
                dataset = dataset.map(
                    self._pre_cache_map, 1 if use_cache else self._num_parallel_calls
                )

                if use_cache:
                    if self._clear_cache:
                        cache_managers.clear()
                    dataset = cache_managers(dataset)
                else:
                    logging.warning(
                        "`pre_cache_map` supplied but no `cache_managers` "
                        "given - not caching."
                    )

            if self._repeats != -1:
                dataset = dataset.repeat(self._repeats)

            if self._shuffle_buffer is not None:
                dataset = dataset.shuffle(self._shuffle_buffer)

            if self._pre_batch_map is not None:
                dataset = dataset.map(self._pre_batch_map, self._num_parallel_calls)

            dataset = dataset.batch(
                self.batch_size, drop_remainder=self._drop_remainder
            )

            if self._post_batch_map is not None:
                dataset = dataset.map(self._post_batch_map, self._num_parallel_calls)

            if self._prefetch_buffer is not None:
                dataset = dataset.prefetch(self._prefetch_buffer)
            return dataset
