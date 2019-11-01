from typing import Any, Callable, TypeVar, Optional, List, Union
from gin.config import parse_config_files_and_bindings
from gin.config import operative_config_str
from gin.config import query_parameter

_FuncT = TypeVar('_FuncT', bound=Callable[..., Any])

# this is the correct one, but the incorrect one below makes things work better
# it requires use as `@configurable()` i.e. it requires the brackets
# but that seems a small price to pay
# def configurable(name_or_fn: Union[Optional[str], _FuncT] = None,
#                  module: Optional[str] = None,
#                  whitelist: Optional[List[str]] = None,
#                  blacklist: Optional[List[str]] = None
#                 ) -> Union[Callable[[_FuncT], _FuncT], _FuncT]:
#     ...


def configurable(name: Optional[str] = None,
                 module: Optional[str] = None,
                 whitelist: Optional[List[str]] = None,
                 blacklist: Optional[List[str]] = None
                ) -> Callable[[_FuncT], _FuncT]:
    ...


def external_configurable(fn_or_cls: _FuncT,
                          name: Optional[str] = None,
                          module: Optional[str] = None,
                          whitelist: Optional[List[str]] = None,
                          blacklist: Optional[List[str]] = None) -> _FuncT:
    ...
