import abc
import contextlib
import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import gin
import tensorflow as tf

from kblocks.functools import get as get_function
from kblocks.functools import serialize_function
from kblocks.serialize import register_serializable


@contextlib.contextmanager
def global_generator_context(rng: tf.random.Generator):
    global_rng = tf.random.get_global_generator()
    try:
        tf.random.set_global_generator(rng)
        yield
    finally:
        tf.random.set_global_generator(global_rng)


def _get_rng(seed: Optional[int] = None) -> tf.random.Generator:
    """Get a `tf.random.Generator` either `from_seed` or `split` from global."""
    with tf.init_scope():
        if seed is None:
            return tf.random.get_global_generator().split(1)[0]
        return tf.random.Generator.from_seed(seed)


class Transform(tf.Module):
    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Transform":
        return cls(**config)

    @abc.abstractmethod
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        raise NotImplementedError("Abstract method")


@gin.configurable(module="kb.data")
@register_serializable
class Take(Transform):
    def __init__(self, count: int):
        self._count = count
        super().__init__()

    @property
    def count(self) -> int:
        return self._count

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.take(self.count)

    def get_config(self) -> Dict[str, Any]:
        return dict(count=self.count)


@gin.configurable(module="kb.data")
@register_serializable
class Repeat(Transform):
    def __init__(self, count: Optional[int] = None):
        self._count = count
        super().__init__()

    @property
    def count(self) -> Optional[int]:
        return self._count

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.repeat(self.count)

    def get_config(self) -> Dict[str, Any]:
        return dict(count=self.count)


def _maybe_ragged_batch(
    dataset: tf.data.Dataset, batch_size: int, drop_remainder: bool = False
):
    kwargs = dict(batch_size=batch_size, drop_remainder=drop_remainder)
    if any(
        (
            len(s.shape) > 0 and s.shape[0] is None
            for s in tf.nest.flatten(dataset.element_spec)
        )
    ):
        # ragged batch
        return dataset.apply(tf.data.experimental.dense_to_ragged_batch(**kwargs))

    return dataset.batch(**kwargs)


@gin.configurable(module="kb.data")
@register_serializable
class Map(Transform):
    def __init__(
        self,
        func: Callable,
        num_parallel_calls: int = 1,
        deterministic: Optional[bool] = None,
        arguments: Optional[Mapping[str, Any]] = None,
    ):
        self._func = get_function(func)
        self._num_parallel_calls = num_parallel_calls
        self._deterministic = deterministic
        self._arguments = {} if arguments is None else dict(arguments)
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(
            func=serialize_function(self._func),
            deterministic=self._deterministic,
            num_parallel_calls=self._num_parallel_calls,
            arguments=self._arguments,
        )

    @property
    def func(self) -> Callable:
        func = self._func
        if self._arguments:
            func = functools.partial(func, **self._arguments)
        return func

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(
            self.func,
            num_parallel_calls=self._num_parallel_calls,
            deterministic=self._deterministic,
        )


@gin.configurable(module="kb.data")
def map_transform(func: Callable, *args, use_rng: bool = False, **kwargs) -> Map:
    return MapRng(func, *args, **kwargs) if use_rng else Map(func, *args, **kwargs)


@gin.configurable(module="kb.data")
@register_serializable
class MapRng(Map):
    """Map where func is called in a `global_generator_context`."""

    def __init__(
        self,
        func: Callable,
        num_parallel_calls: int = 1,
        deterministic: Optional[bool] = None,
        arguments: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            func=func,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
            arguments=arguments,
        )
        self._seed = seed
        self._rng = None

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["seed"] = self._seed
        return config

    @property
    def func(self) -> Callable:
        func = super().func

        @functools.wraps(func)
        def ret_func(*args, **kwargs):
            if self._rng is None:
                self._rng = _get_rng(self._seed)
            with global_generator_context(self._rng):
                return func(*args, **kwargs)

        return ret_func


@gin.configurable(module="kb.data")
@register_serializable
class Enumerate(Transform):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.enumerate()


@gin.configurable(module="kb.data")
@register_serializable
class Batch(Transform):
    def __init__(self, batch_size: int, drop_remainder: bool = False):
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(batch_size=self._batch_size, drop_remainder=self._drop_remainder)

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.batch(
            batch_size=self._batch_size, drop_remainder=self._drop_remainder
        )


@gin.configurable(module="kb.data")
@register_serializable
class DenseToRaggedBatch(Batch):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(
                batch_size=self._batch_size, drop_remainder=self._drop_remainder
            )
        )


@gin.configurable(module="kb.data")
@register_serializable
class DenseToSparseBatch(Transform):
    def __init__(self, batch_size: int, row_shape: Iterable[int]):
        self._batch_size = batch_size
        self._row_shape = tuple(row_shape)
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(batch_size=self._batch_size, row_shape=self._row_shape)

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.apply(
            tf.data.experimental.dense_to_sparse_batch(
                batch_size=self._batch_size, row_shape=self._row_shape
            )
        )


@gin.configurable(module="kb.data")
@register_serializable
class Unbatch(Transform):
    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.unbatch()


@gin.configurable(module="kb.data")
@register_serializable
class Prefetch(Transform):
    def __init__(self, buffer_size: int = 1):
        self._buffer_size = buffer_size
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(buffer_size=self._buffer_size)

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.prefetch(self._buffer_size)


@gin.configurable(module="kb.data")
@register_serializable
class Shuffle(Transform):
    def __init__(
        self,
        buffer_size: int,
        seed: Optional[int] = None,
        reshuffle_each_iteration: Optional[bool] = None,
    ):
        self._buffer_size = buffer_size
        self._seed = seed
        self._reshuffle_each_iteration = reshuffle_each_iteration
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(
            buffer_size=self._buffer_size,
            seed=self._seed,
            reshuffle_each_iteration=self._reshuffle_each_iteration,
        )

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.shuffle(
            self._buffer_size if self._buffer_size > 0 else len(dataset),
            seed=self._seed,
            reshuffle_each_iteration=self._reshuffle_each_iteration,
        )


@gin.configurable(module="kb.data")
def shuffle(*args, use_rng: bool = False, **kwargs):
    return ShuffleRng(*args, **kwargs) if use_rng else Shuffle(*args, **kwargs)


@gin.configurable(module="kb.data")
@register_serializable
class ShuffleRng(Transform):
    def __init__(self, buffer_size: int, seed: Optional[int] = None):
        self._buffer_size = buffer_size
        self._seed = seed
        self._rng = None
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(buffer_size=self._buffer_size, seed=self._seed)

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def map_func(*args):
            if len(args) == 1:
                (args,) = args
            if self._rng is None:
                self._rng = _get_rng(self._seed)
            size = tf.shape(tf.nest.flatten(args)[0])[0]
            u = self._rng.uniform((size,))
            perm = tf.argsort(u)
            args = tf.nest.map_structure(lambda x: tf.gather(x, perm, axis=0), args)
            return args

        buffer_size = self._buffer_size if self._buffer_size > 0 else len(dataset)
        shuffled = (
            _maybe_ragged_batch(dataset, buffer_size, drop_remainder=False)
            .map(map_func)
            .unbatch()
        )
        cardinality = dataset.cardinality()  # will not change with the following
        if cardinality != shuffled.cardinality():
            shuffled = shuffled.apply(
                tf.data.experimental.assert_cardinality(cardinality)
            )
        return shuffled


def get(identifier) -> Transform:
    if isinstance(identifier, Transform):
        return identifier
    transform = tf.keras.utils.deserialize_keras_object(identifier)
    if not isinstance(transform, Transform):
        raise TypeError(
            f"Deserialized identifier should be a Transform, got {transform}"
        )
    return transform
