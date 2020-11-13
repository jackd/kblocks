"""Path utilities."""
import os

import gin


@gin.configurable(module="os.path")
def join(a: str, p: str) -> str:
    """Configurable equivalent to `os.path.join`. Only accepts 2 args."""
    return os.path.join(a, p)


gin.register(os.path.expanduser, module="os.path")
gin.register(os.path.expandvars, module="os.path")


@gin.configurable(module="kb.path")
def expand(path):
    return os.path.expanduser(os.path.expandvars(path))


@gin.configurable(module="kb.path")
def run_subdir(run: int = 0):
    return f"run-{run:02d}"
