import abc
import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from kblocks.data import transforms
from kblocks.serialize import register_serializable
from kblocks.utils import memoized_property


def applies(transform_func: Callable):
    """
    Wrapper to create transform wrapper as a source.

    Note the returned function only passes `*args` and `**kwargs` passed to it onto
    `transform_func`. If `transform_func` is configured via `gin` then the returned
    function will be configured in the same way.
    """

    def wrapped(f):
        def ret_f(self, *args, **kwargs):
            return self.apply(transform_func(*args, **kwargs))

        for attr in "__name__", "__qualname__":
            setattr(ret_f, attr, getattr(f, attr))
        ret_f.__doc__ = (
            "Get a source with "
            f"`{transform_func.__module__}.{transform_func.__name__}` applied."
        )
        # get parameters (including self)
        params = [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *inspect.signature(transform_func).parameters.values(),
        ]
        ret_f.__signature__ = inspect.Signature(
            params, return_annotation="TransformedSource"
        )
        return ret_f

    return wrapped


class DataSource(tf.Module):
    """
    Serializable/tf.Module wrapper around `tf.data.Dataset` API.

    Unlike `tf.data.Dataset`s, these are can used in conjunction with transforms that
    have state, e.g. `ShuffleRng` or `MapRng` for deterministic and preemptible data
    pipelines.
    """

    @abc.abstractproperty
    def dataset(self):
        raise NotImplementedError("Abstract property")

    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DataSource":
        return cls(**config)

    def apply(self, transform: transforms.Transform) -> "TransformedSource":
        return TransformedSource(self, transform)

    @applies(transforms.Take)
    def take(self, *args, **kwargs):
        pass

    @applies(transforms.Repeat)
    def repeat(self, *args, **kwargs):
        pass

    @applies(transforms.Map)
    def map(self, *args, **kwargs):
        pass

    @applies(transforms.MapRng)
    def map_rng(self, *args, **kwargs):
        pass

    @applies(transforms.Enumerate)
    def enumerate(self):
        pass

    @applies(transforms.Cache)
    def cache(self, *args, **kwargs):
        pass

    @applies(transforms.Batch)
    def batch(self, *args, **kwargs):
        pass

    @applies(transforms.DenseToRaggedBatch)
    def batch_ragged(self, *args, **kwargs):
        pass

    @applies(transforms.DenseToSparseBatch)
    def batch_sparse(self, *args, **kwargs):
        pass

    @applies(transforms.Prefetch)
    def prefetch(self, *args, **kwargs):
        pass

    @applies(transforms.Shuffle)
    def shuffle(self, *args, **kwargs):
        pass

    @applies(transforms.ShuffleRng)
    def shuffle_rng(self, *args, **kwargs):
        pass


@gin.register(module="kb.data")
@register_serializable
class TransformedSource(DataSource):
    def __init__(self, base: DataSource, transform: transforms.Transform):
        self._base = get(base)
        self._transform = transforms.get(transform)
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(
            base=tf.keras.utils.serialize_keras_object(self._base),
            transform=tf.keras.utils.serialize_keras_object(self._transform),
        )

    @memoized_property
    def dataset(self):
        return self._base.dataset.apply(self._transform)


@gin.register(module="kb.data")
@register_serializable
class RangeSource(DataSource):
    def __init__(self, start, stop: Optional = None, step: Optional = None):
        self._start = start
        self._stop = stop
        self._step = step
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(start=self._start, stop=self._stop, step=self._step)

    @memoized_property
    def dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.range(self._start, self._stop, self._step)


@gin.register(module="kb.data")
@register_serializable
class FunctionSource(DataSource):
    def __init__(self, func: Callable, arguments: Optional[Mapping[str, Any]] = None):
        if not callable(func):
            func = tf.keras.utils.deserialize_keras_object(func)
        if not callable(func):
            raise ValueError(
                f"`func` must be callable or deserializable to a callable, got {func}"
            )
        self._func = func
        self._arguments = dict(arguments) if arguments else {}
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return dict(
            func=tf.keras.utils.serialize_keras_object(self._func),
            arguments=self._arguments,
        )

    @memoized_property
    def dataset(self) -> tf.data.Dataset:
        return self._func(**self._arguments)


@gin.register(module="kb.data")
@register_serializable
class TfdsSource(FunctionSource):
    def __init__(self, name: str, split: str, shuffle_files: bool = False, **kwargs):

        super().__init__(
            tfds.load,
            arguments=dict(
                name=name, split=split, shuffle_files=shuffle_files, **kwargs,
            ),
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(self._arguments)


@register_serializable
class DelegatingSource(DataSource):
    def __init__(self, source: DataSource):
        self._source = get(source)
        super().__init__()

    @property
    def dataset(self):
        return self._source.dataset

    def get_config(self):
        return dict(source=tf.keras.utils.serialize_keras_object(self._source))


@gin.register(module="kb.data")
def apply(
    source: DataSource,
    transform: Union[transforms.Transform, Iterable[Optional[transforms.Transform]]],
):
    if isinstance(transform, transforms.Transform):
        return TransformedSource(source, transform)
    for t in transform:
        if t is not None:
            source = TransformedSource(source, t)
    return source


def get(identifier) -> DataSource:
    if isinstance(identifier, DataSource):
        return identifier
    out = tf.keras.utils.deserialize_keras_object(identifier)
    if not isinstance(out, DataSource):
        raise TypeError(f"Deserialized identifier should be a DataSource, got {out}")
    return out
