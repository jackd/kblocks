from typing import Optional

import gin
import tensorflow as tf


@gin.configurable(module="kb.random")
def generator_from_seed(seed: int, alg: Optional[str] = None) -> tf.random.Generator:
    return tf.random.Generator.from_seed(seed, alg=alg)


@gin.configurable(module="kb.random")
def generator_from_non_deterministic_state(
    alg: Optional[str] = None,
) -> tf.random.Generator:
    return tf.random.Generator.from_non_deterministic_state(alg=alg)


for src, dst in (
    (tf.random.Generator.from_seed, generator_from_seed),
    (
        tf.random.Generator.from_non_deterministic_state,
        generator_from_non_deterministic_state,
    ),
):
    dst.__doc__ = src.__doc__

del src, dst
