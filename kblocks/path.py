"""Path utilities."""
import os
import tempfile
import uuid
from typing import Optional

import gin
import tensorflow as tf


@gin.configurable(module="os.path")
def join(a: str, p: str) -> str:
    """Configurable equivalent to `os.path.join`. Only accepts 2 args."""
    return os.path.join(a, p)


gin.register(os.path.expanduser, module="os.path")
gin.register(os.path.expandvars, module="os.path")


@gin.register(module="kb.path")
def expand(path):
    return os.path.expanduser(os.path.expandvars(path))


@gin.configurable(module="kb.path")
def run_subdir(run: int = 0):
    return f"run-{run:02d}"


@gin.register(module="kb.path")
def temp_dir(subdir: Optional[str] = "kblocks"):
    """
    Get the path to a non-existent subdirectory of system temporary directory.

    The directory is guaranteed not to exist when the function returns. The user is
    responsible for optionally creating and/or deleting it.
    """
    tmp = tempfile.gettempdir()
    if subdir is not None:
        tmp = os.path.join(tmp, subdir)

    def get_dir():
        return os.path.join(tmp, str(uuid.uuid4()))

    experiment_dir = get_dir()
    while tf.io.gfile.isdir(experiment_dir):
        experiment_dir = get_dir()
    return experiment_dir
