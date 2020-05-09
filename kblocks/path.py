"""Path utilities."""
import os

import gin


def _expand(path):
    return os.path.expanduser(os.path.expandvars(path))


@gin.configurable(module="kb.path")
def join(a: str, p: str) -> str:
    """Configurable equivalent to `os.path.join`."""
    return os.path.join(a, p)


@gin.configurable(module="kb.path")
def log_dir(root_dir: str = "~/kblocks") -> str:
    return _expand(os.path.join(root_dir, "logs"))


@gin.configurable(module="kb.path")
def model_dir(
    root_dir: str = "~/kblocks",
    problem_id: str = "default_prob",
    model_id: str = "default_model",
    variant_id: str = "v0",
    run: int = 0,
) -> str:
    """Good default configurable model directory."""
    args = [k for k in (root_dir, problem_id, model_id, variant_id) if k is not None]
    if run is not None:
        args.append("run-{:02d}".format(run))
    return _expand(os.path.expandvars(os.path.join(*args)))
