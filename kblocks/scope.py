from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Scope(Generic[T]):
    def __init__(self, default: Optional[T] = None, name: str = "scope"):
        self._stack = []
        self._name = name
        if default is not None:
            self._stack.append(Scoped(self, default))

    def __call__(self, instance: T):
        scoped = Scoped[T](self, instance)
        self._stack.append(scoped)
        return scoped

    def get_default(self):
        if len(self._stack) == 0:
            raise ValueError(
                "Cannot get default value - {} stack empty".format(self._name)
            )
        return self._stack[-1].instance


class Scoped(Generic[T]):
    def __init__(self, scope: Scope[T], instance: T):
        self._scope = scope
        self._instance = instance

    @property
    def instance(self):
        return self._instance

    @property
    def scope(self):
        return self._scope

    def __enter__(self):
        self._scope._stack.append(self)
        return self._instance

    def __exit__(self, type, value, traceback):  # pylint:disable=redefined-builtin
        out = self._scope._stack.pop()
        assert out is self
