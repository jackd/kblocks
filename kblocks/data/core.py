from typing import Callable, Iterable, Optional, Union

import gin
import tensorflow as tf
import tfrng

from kblocks.path import expand

Transform = Callable[[tf.data.Dataset], tf.data.Dataset]


@gin.configurable(module="kb.data")
def apply(
    dataset: tf.data.Dataset,
    transform: Union[Transform, Iterable[Optional[Transform]]],
) -> tf.data.Dataset:
    if callable(transform):
        return dataset.apply(transform)

    for tr in transform:
        if tr is not None:
            dataset = dataset.apply(tr)
    return dataset


@gin.configurable(module="kb.data")
def batch(batch_size: int, drop_remainder: bool = False) -> Transform:
    def transform(dataset: tf.data.Dataset):
        return dataset.batch(batch_size, drop_remainder=drop_remainder)

    return transform


@gin.configurable(module="kb.data")
def compound_transform(transforms: Iterable[Optional[Transform]]):
    transforms = [t for t in transforms if t is not None]
    if len(transforms) == 1:
        return transforms[0]

    def transform(dataset):
        for t in transforms:
            dataset = t(dataset)
        return dataset

    return transform


@gin.configurable(module="kb.data")
def cache(filename="") -> Transform:
    filename = expand(filename)

    def transform(dataset: tf.data.Dataset):
        return dataset.cache(filename)

    return transform


@gin.configurable(module="kb.data")
def enumerate_transform() -> Transform:
    def transform(dataset):
        return dataset.enumerate()

    return transform


@gin.configurable(module="kb.data")
def filter_transform(predicate: Callable) -> Transform:
    def transform(dataset):
        return dataset.filter(predicate)

    return transform


@gin.configurable(module="kb.data")
def map_transform(
    map_func: Callable,
    num_parallel_calls: Optional[int] = None,
    deterministic: Optional[bool] = None,
) -> Transform:
    def transform(dataset):
        return dataset.map(
            map_func=map_func,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
        )

    return transform


@gin.configurable(module="kb.data")
def prefetch(buffer_size: int) -> Transform:
    def transform(dataset):
        return dataset.prefetch(buffer_size=buffer_size)

    return transform


@gin.configurable(module="kb.data")
def shuffle(
    buffer_size: int,
    seed: Optional[int] = None,
    reshuffle_each_iteration: Optional[bool] = None,
) -> Transform:
    def transform(dataset):
        return dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )

    return transform


@gin.configurable(module="kb.data")
def skip(count: int) -> Transform:
    def transform(dataset):
        return dataset.skip(count)

    return transform


@gin.configurable(module="kb.data")
def take(count: int) -> Transform:
    def transform(dataset):
        return dataset.take(count)

    return transform


@gin.configurable(module="kb.data")
def unbatch() -> Transform:
    def transform(dataset):
        return dataset.unbatch()

    return transform


@gin.configurable(module="kb.data")
def repeat(count: Optional[int] = None) -> Transform:
    def transform(dataset):
        return dataset.repeat(count)

    return transform


@gin.configurable(module="kb.data")
def with_options(options: tf.data.Options) -> Transform:
    def transform(dataset):
        return dataset.with_options(options)

    return transform


@gin.configurable(module="kb.data")
def options(**kwargs):
    options = tf.data.Options()
    for k, v in kwargs.items():
        setattr(options, k, v)
    return options


# experimental

dense_to_ragged_batch = gin.external_configurable(
    tf.data.experimental.dense_to_ragged_batch, module="kb.data"
)
dense_to_sparse_batch = gin.external_configurable(
    tf.data.experimental.dense_to_sparse_batch, module="kb.data"
)

snapshot = gin.external_configurable(tf.data.experimental.snapshot, module="kb.data")

assert_cardinality = gin.external_configurable(
    tf.data.experimental.assert_cardinality, module="kb.data"
)

# tfrng
stateless_map = gin.external_configurable(tfrng.data.stateless_map, module="tfrng.data")
generator_map = gin.external_configurable(tfrng.data.generator_map, module="tfrng.data")
