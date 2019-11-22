from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Optional
import gin


@gin.configurable(module='kb.framework')
def steps_in_examples(num_examples: int, batch_size: Optional[int] = None):
    return num_examples // get_batch_size(batch_size)


@gin.configurable(module='kb.framework')
def get_batch_size(batch_size: Optional[int] = None) -> int:
    if batch_size is None:
        bs = gin.query_parameter('batch_size/macro.value')
        if bs is None:
            raise RuntimeError('No value assigned to "batch_size"')
        return bs
    else:
        return batch_size


@gin.configurable(module='kb.framework')
def get_epochs(epochs: Optional[int] = None) -> int:
    if epochs is None:
        ep = gin.query_parameter('epochs/macro.value')
        if ep is None:
            raise RuntimeError('No value assigned to "epochs"')
        return ep
    else:
        return epochs


@gin.configurable(module='kb.framework')
def get_examples_per_epoch(examples_per_epoch: Optional[int] = None) -> int:
    if examples_per_epoch is None:
        from kblocks.framework.problems.core import Problem
        return Problem.default().examples_per_epoch('train')
    return examples_per_epoch


@gin.configurable(module='kb.framework')
def steps_per_epoch(batch_size: Optional[int] = None,
                    examples_per_epoch: Optional[int] = None) -> int:
    return get_examples_per_epoch(examples_per_epoch) // get_batch_size(
        batch_size)


@gin.configurable(module='kb.framework')
def steps_in_epochs(epochs: int,
                    batch_size: Optional[int] = None,
                    examples_per_epoch: Optional[int] = None) -> int:
    return steps_per_epoch(batch_size, examples_per_epoch) * epochs


@gin.configurable(module='kb.framework')
def epochs_in_steps(steps: int,
                    batch_size: Optional[int] = None,
                    examples_per_epoch: Optional[int] = None) -> int:
    return steps * get_batch_size(batch_size) // get_examples_per_epoch(
        examples_per_epoch)


get_step = gin.external_configurable(tf.summary.experimental.get_step,
                                     module='kb.framework')
set_step = tf.summary.experimental.set_step
