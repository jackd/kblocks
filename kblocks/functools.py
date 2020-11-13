import functools
import importlib
from typing import Callable, Mapping, Union

import gin
import tensorflow as tf

from kblocks.serialize import register_serializable


@gin.register(module="kb.functools")
@register_serializable
class Function:
    def __init__(self, func: Union[Callable, Mapping]):
        if not callable(func):
            func = getattr(importlib.import_module(func["module"]), func["name"])
            assert callable(func)
        elif func.__name__ == "<lambda>":
            raise ValueError("Cannot wrap `lambda` in `kblocks.functools.Function`.")
        self._func = func

    def get_config(self):
        return dict(name=self._func.__name__, module=self._func.__module__)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


@gin.register(module="functools")
@tf.keras.utils.register_keras_serializable("functools")
class partial:
    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.keywords = kwargs

    def __call__(self, *args, **kwargs):
        return functools.partial(self.func, *self.args, **self.keywords)(
            *args, **kwargs
        )

    def get_config(self):
        return dict(
            func=serialize_function(self.func),
            args=list(self.args),
            keywords=dict(self.keywords),
        )

    @classmethod
    def from_config(cls, config):
        return cls(get(config["func"]), *config["args"], **config["keywords"])


def serialize_function(func: Callable):
    if hasattr(func, "get_config"):
        keras_obj = func
    elif isinstance(func, functools.partial):
        keras_obj = partial(func.func, *func.args, **func.keywords)
    else:
        keras_obj = Function(func)
    return tf.keras.utils.serialize_keras_object(keras_obj)


def get(identifier):
    if callable(identifier):
        return identifier
    func = tf.keras.utils.deserialize_keras_object(identifier)
    if not callable(func):
        raise ValueError(f"Invalid func {func}")
    return func
