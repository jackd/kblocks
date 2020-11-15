"""Configurable utilities for ray."""
import os
from typing import Optional

import gin
import ray

DEFAULT = "__default__"


@gin.configurable(module="kb.utils")
def ray_init(
    redis_address: Optional[str] = DEFAULT,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    local_mode: bool = False,
    **kwargs
):
    if redis_address == DEFAULT:
        redis_address = os.environ.get("REDIS_ADDRESS")
    return ray.init(
        redis_address,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=local_mode,
        **kwargs
    )
