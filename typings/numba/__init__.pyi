import numpy as np
from typing import TypeVar, Callable, Any, Iterable, Optional, Tuple, List

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


def prange(start_or_stop: int,
           stop: Optional[int] = None,
           stride: Optional[int] = None) -> Iterable[int]:
    ...


def jitclass(spec: List[Tuple[str, Any]]) -> Callable[[_FuncT], _FuncT]:
    ...


__version__: str
uint8: Any
uint32: Any
int64: Any
float32: Any
float64: Any
void: Any


def from_type(dtype: np.dtype) -> Any:
    ...
