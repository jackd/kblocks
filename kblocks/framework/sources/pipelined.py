from typing import Any, Callable, Mapping, Optional, Union

import gin
import tensorflow as tf
from absl import logging

from kblocks.extras.cache import CacheManager
from kblocks.framework.sources import core
from kblocks.utils import memoized_property

AUTOTUNE = tf.data.experimental.AUTOTUNE


@gin.configurable(module="kb.framework")
def batch_dataset(
    dataset: tf.data.Dataset, batch_size: int, drop_remainder: bool = False
) -> tf.data.Dataset:
    return dataset.batch(batch_size, drop_remainder)


@gin.configurable(module="kb.framework")
class PipelinedSource(core.DataSource):
    def __init__(
        self,
        source: core.DataSource,
        batch_fn: Callable[[tf.data.Dataset], tf.data.Dataset],
        pre_cache_map: Optional[Callable] = None,
        pre_batch_map: Optional[Callable] = None,
        post_batch_map: Optional[Callable] = None,
        cache_managers: Optional[Mapping[str, CacheManager]] = None,
        shuffle_buffer: Optional[int] = None,
        prefetch_buffer: int = AUTOTUNE,
        num_parallel_calls: int = AUTOTUNE,
        clear_cache: bool = False,
        meta: Optional[Mapping[str, Any]] = None,
        epoch_lengths: Optional[Union[Callable, Mapping[core.Split, int]]] = None,
    ):
        assert isinstance(source, core.DataSource)
        self._base_source = source
        self._batch_fn = batch_fn
        self._pre_cache_map = pre_cache_map
        self._pre_batch_map = pre_batch_map
        self._post_batch_map = post_batch_map
        self._cache_managers = cache_managers
        self._shuffle_buffer = shuffle_buffer
        self._prefetch_buffer = prefetch_buffer
        self._num_parallel_calls = num_parallel_calls
        self._clear_cache = clear_cache
        self._epoch_lengths = epoch_lengths
        if meta is None:
            meta = source.meta
        self._meta = meta

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta

    def get_dataset(self, split: core.Split):
        dataset = self._base_source.get_dataset(split)
        with tf.keras.backend.learning_phase_scope(split == "train"):
            cache_manager = (
                None
                if self._cache_managers is None
                else self._cache_managers.get(split)
            )
            if self._pre_cache_map is not None:
                dataset = dataset.map(
                    self._pre_cache_map,
                    self._num_parallel_calls if cache_manager is None else 1,
                )
            if cache_manager is not None:
                if self._clear_cache:
                    cache_manager.clear()
                dataset = cache_manager(dataset)
            else:
                logging.warning(
                    "`pre_cache_map` supplied byt no `cache_manager` - not caching."
                )

            if self._shuffle_buffer is not None:
                dataset = dataset.shuffle(self._shuffle_buffer)
            if self._pre_batch_map is not None:
                dataset = dataset.map(self._pre_batch_map, self._num_parallel_calls)

            dataset = self._batch_fn(dataset)
            if self._post_batch_map is not None:
                dataset = dataset.map(self._post_batch_map, self._num_parallel_calls)
            if self._prefetch_buffer is not None:
                dataset = dataset.prefetch(self._prefetch_buffer)
        return dataset

    def epoch_length(self, split: core.Split) -> Optional[int]:
        return (
            None
            if self._epoch_lengths is None
            else self._epoch_lengths(split)
            if callable(self._epoch_lengths)
            else self._epoch_lengths.get(split)
        )

    @memoized_property
    def element_spec(self):
        def always_throws():
            raise NotImplementedError("Should not be called")

        base_spec = self._base_source.element_spec
        dataset = tf.data.Dataset.from_generator(
            always_throws,
            output_types=tf.nest.map_structure(lambda x: x.dtype, base_spec),
            output_shapes=tf.nest.map_structure(lambda x: x.shape, base_spec),
        )
        for m in self._pre_cache_map, self._pre_batch_map:
            if m is not None:
                dataset = dataset.map(m)
        dataset = self._batch_fn(dataset)
        if self._post_batch_map is not None:
            dataset = dataset.map(self._post_batch_map)
        return dataset.element_spec
