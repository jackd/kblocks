import os
from typing import Callable, Optional, Tuple

import gin
import tensorflow as tf
from absl import logging

import kblocks.data.tfrecords as tfrecords_lib
from kblocks.data.core import Transform, cache


@gin.configurable(module="kb.data")
def save_load_cache(
    path: str,
    compression: Optional[str] = None,
    shard_func: Optional[Callable] = None,
    reader_func: Optional[Callable] = None,
) -> Transform:
    """Cache transform that uses `tf.data.experimental.[save,load]`."""

    def transform(dataset):
        tf.io.gfile.makedirs(path)
        matching = tf.io.matching_files(path + "/*")
        if tf.shape(matching)[0] == 0:
            if tf.executing_eagerly():
                logging.info(f"Saving dataset to {path}")
            try:
                tf.data.experimental.save(
                    dataset, path=path, compression=compression, shard_func=shard_func
                )
            except:
                # remove partially created save
                tf.io.gfile.rmtree(path)
                raise
        return tf.data.experimental.load(
            path=path,
            element_spec=dataset.element_spec,
            compression=compression,
            reader_func=reader_func,
        ).apply(tf.data.experimental.assert_cardinality(dataset.cardinality()))

    return transform


@gin.configurable(module="kb.data")
def tfrecords_cache(path: str, compression: Optional[str] = None):
    def transform(dataset):
        return tfrecords_lib.tfrecords_cache(
            dataset, cache_dir=path, compression=compression
        )

    return transform


def _repeated_paths(path: str, num_repeats: int) -> Tuple[str, ...]:
    if num_repeats > 10000:
        raise ValueError("Can only handle up to 1e4 repeats")
    return tuple(os.path.join(path, f"repeat-{i:04d}") for i in range(num_repeats))


@gin.configurable(module="kb.data")
def repeated_cache(
    path: str,
    num_repeats: int,
    cache_factory: Callable[[str], Transform] = cache,
    transform: Transform = lambda x: x,
) -> Transform:
    """
    Get a transform for caching multpile epochs of the underlying dataset.

    This is necessary for caching after data augmentation.

    The output of the return transform calls have `num_repeats` times as many elements
    the input dataset.
    """
    paths = _repeated_paths(path=path, num_repeats=num_repeats)

    def ret_transform(dataset):
        # possibly create cached files in eager mode
        for p in paths:
            cache_factory(p)(dataset)

        return tf.data.Dataset.from_tensor_slices(paths).flat_map(
            lambda p_: transform(cache_factory(p_)(dataset))
        )

    return ret_transform


@gin.configurable(module="kb.data")
def random_repeated_cache(
    path: str,
    num_repeats: int,
    cache_factory: Callable[[str], Transform] = cache,
    shuffle_seed: Optional[int] = None,
    reshuffle_each_iteration: bool = True,
    transform: Transform = lambda x: x,
) -> Transform:
    """
    Get a transform for caching multiple epochs of the underlying dataset.

    This is necessary for caching after data augmentation.

    Each iteration returns a random repeat. The output of the returned transform calls
    have the same number of elements as the input dataset.
    """
    paths = _repeated_paths(path=path, num_repeats=num_repeats)

    def ret_transform(dataset):
        # possibly create cached files in eager mode
        for p in paths:
            cache_factory(p)(dataset)
        path_dataset = tf.data.Dataset.from_tensor_slices(path).shuffle(
            len(paths),
            seed=shuffle_seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
        return transform(
            path_dataset.take(1).flat_map(lambda p_: cache_factory(p_)(dataset))
        )

    return ret_transform
