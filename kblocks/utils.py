"""Utilities used throughout kblocks."""
import inspect
from typing import Any, Callable, Iterable, List, Optional, TypeVar

import gin
import setproctitle

T = TypeVar("T")


@gin.configurable(module="kb.utils")
def proc(title: str = "kblocks"):
    setproctitle.setproctitle(title)


# utility function registration


@gin.register(module="kb.utils")
def identity(x):
    return x


@gin.register(module="kb.utils")
def concat(a: Iterable[T], b: Iterable[T]) -> List[T]:
    out = list(a)
    out.extend(b)
    return out


gin.register(dict, module="kb.utils")


@gin.register(name_or_fn="getattr", module="kb.utils")
def _getattr(object, name: str, default=None):  # pylint: disable=redefined-builtin
    return getattr(object, name, default)


@gin.register(module="kb.utils")
def call(func: Callable, **kwargs):
    """Configurable version of `func(**kwargs)`."""
    return func(**kwargs)


class memoized_property(property):  # pylint: disable=invalid-name
    """Descriptor that mimics @property but caches output in member variable."""

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        # See https://docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            # cached = self.fget(obj)
            cached = super(memoized_property, self).__get__(obj, type)
            setattr(obj, attr, cached)
        return cached


def delegates(to=None, keep=False):
    """Decorator: replace `**kwargs` in signature with params from `to`"""

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f
