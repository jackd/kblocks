from typing import TypeVar, Callable, Any

_FuncT = TypeVar('_FuncT', bound=Callable[..., Any])


def njit(*args, **kwargs) -> Callable[[_FuncT], _FuncT]:
    ...


def jit(signature_or_function=None,
        locals={},
        target='cpu',
        cache=False,
        pipeline_class=None,
        **options) -> Callable[[_FuncT], _FuncT]:
    ...
