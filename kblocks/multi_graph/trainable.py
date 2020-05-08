import functools
from typing import Callable, Optional

import gin
import tensorflow as tf
from absl import logging

from kblocks import multi_graph as mg
from kblocks.framework.pipelines import BasePipeline
from kblocks.framework.sources import DataSource, PipelinedSource
from kblocks.framework.trainable import Trainable


@gin.configurable(module="kb.framework")
def multi_graph_trainable(
    build_fn: Callable,
    base_source: DataSource,
    batch_size: int,
    compiler: Callable[[tf.keras.Model], None],
    model_dir: Optional[str] = None,
    build_with_batch_size: bool = True,
    #   rebuild_model_with_xla: bool = False,
    use_model_builders: bool = False,
    **pipeline_kwargs
):
    logging.info("Building multi graph...")
    built = mg.build_multi_graph(
        functools.partial(build_fn, **base_source.meta),
        base_source.example_spec,
        batch_size if build_with_batch_size else None,
        use_model_builders=use_model_builders,
    )

    logging.info("Successfully built!")

    pipeline = BasePipeline(
        batch_size,
        pre_cache_map=built.pre_cache_map,
        pre_batch_map=built.pre_batch_map,
        post_batch_map=built.post_batch_map,
        **pipeline_kwargs
    )
    source = PipelinedSource(base_source, pipeline)
    model = built.trained_model
    # if rebuild_model_with_xla:
    #     with tf.xla.experimental.jit_scope():
    #         model = tf.keras.models.clone_model(model)

    if compiler is not None:
        compiler(model)
    return Trainable(source, model, model_dir)
