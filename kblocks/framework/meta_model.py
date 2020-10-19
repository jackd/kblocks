import functools
from typing import Callable, Optional

import gin
import tensorflow as tf
from absl import logging

from kblocks.framework.batchers import Batcher
from kblocks.framework.sources import DataSource, PipelinedSource
from kblocks.framework.trainable import Trainable
from meta_model import pipeline as pl


@gin.configurable(module="kb.framework")
def meta_model_trainable(
    build_fn: Callable,
    base_source: DataSource,
    batcher: Batcher,
    compiler: Callable[[tf.keras.Model], None],
    model_dir: Optional[str] = None,
    rebuild_model_with_xla: bool = False,
    **pipeline_kwargs
):
    logging.info("Building pipelined model...")
    pipeline, model = pl.build_pipelined_model(
        functools.partial(build_fn, **base_source.meta),
        element_spec=base_source.element_spec,
        batcher=batcher,
    )

    logging.info("Pipelined model built!")

    source = PipelinedSource(
        base_source,
        batcher=batcher,
        pre_cache_map=pipeline.pre_cache_map,
        pre_batch_map=pipeline.pre_batch_map,
        post_batch_map=pipeline.post_batch_map,
        **pipeline_kwargs
    )
    if rebuild_model_with_xla:
        with tf.xla.experimental.jit_scope():
            model = tf.keras.models.clone_model(model)

    if compiler is not None:
        compiler(model)
    return Trainable(source, model, model_dir)
