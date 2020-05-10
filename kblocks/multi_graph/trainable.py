import functools
from typing import Callable, Optional

import gin
import tensorflow as tf
from absl import logging

from kblocks import multi_graph as mg
from kblocks.framework.sources import DataSource
from kblocks.framework.sources.pipelined import PipelinedSource, batch_dataset
from kblocks.framework.trainable import Trainable


@gin.configurable(module="kb.framework")
def multi_graph_trainable(
    build_fn: Callable,
    base_source: DataSource,
    batch_size: int,
    compiler: Callable[[tf.keras.Model], None],
    model_dir: Optional[str] = None,
    build_with_batch_size: bool = True,
    use_model_builders: bool = False,
    **pipeline_kwargs
):
    logging.info("Building multi graph...")
    built = mg.build_multi_graph(
        functools.partial(build_fn, **base_source.meta),
        base_source.element_spec,
        batch_size if build_with_batch_size else None,
        use_model_builders=use_model_builders,
    )

    logging.info("Successfully built!")

    source = PipelinedSource(
        base_source,
        batch_fn=functools.partial(batch_dataset, batch_size=batch_size),
        pre_cache_map=built.pre_cache_map,
        pre_batch_map=built.pre_batch_map,
        post_batch_map=built.post_batch_map,
        **pipeline_kwargs
    )
    model = built.trained_model
    if compiler is not None:
        compiler(model)
    return Trainable(source, model, model_dir)
