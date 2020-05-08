import imp
from types import ModuleType
from typing import Any, Iterable, Tuple

import gin

BLACKLIST = ("serialize", "deserialize", "get")


def wrapped_items(
    src_module: ModuleType, gin_module: str, blacklist: Iterable[str] = BLACKLIST,
) -> Iterable[Tuple[str, Any]]:
    for k in dir(src_module):
        v = getattr(src_module, k)
        if k not in blacklist and callable(v):
            yield (k, gin.external_configurable(v, name=k, module=gin_module))


def wrapped_module(
    dst_name,
    src_module: ModuleType,
    gin_module: str,
    blacklist: Iterable[str] = BLACKLIST,
) -> ModuleType:
    mod = imp.new_module(dst_name)
    for k, v in wrapped_items(src_module, gin_module, blacklist):
        setattr(mod, k, v)
    return mod


def renamed(child_mod: ModuleType, root_mod: ModuleType, new_root: str):
    root_name = root_mod.__name__
    child_name = child_mod.__name__
    if not child_name.startswith(root_name):
        raise ValueError(
            "child_name should start with root_name, but {} does not start "
            "with {}".format(child_name, root_name)
        )
    return "{}{}".format(new_root, child_name[len(root_name) :])
