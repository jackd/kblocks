import functools
from typing import Callable, Optional, Sequence

import gin
import tensorflow as tf

from kblocks.data import sources, transforms
from kblocks.trainables.core import Trainable
from meta_model import pipeline as pl


@gin.configurable(module="kb.trainable")
def build_meta_model_trainable(
    meta_model_fn: Callable,
    batcher: transforms.Transform,
    train_source: sources.DataSource,
    validation_source: Optional[sources.DataSource],
    shuffle_buffer: int = 1024,
    train_cache: Optional[transforms.Transform] = None,
    validation_cache: Optional[transforms.Transform] = None,
    loss: Optional[tf.keras.losses.Loss] = None,
    metrics: Optional = None,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    use_rngs: bool = False,
    callbacks: Sequence[tf.keras.callbacks.Callback] = (),
    name: Optional[str] = None,
):
    pipeline, model = pl.build_pipelined_model(
        meta_model_fn, train_source.dataset.element_spec, batcher
    )
    if optimizer is None:
        assert loss is None
        assert metrics is None
    else:
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    map_transform = functools.partial(transforms.map_transform, use_rng=use_rngs)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_source = sources.apply(
        train_source,
        [
            map_transform(pipeline.pre_cache_map_funcs(training=True)),
            train_cache,
            transforms.shuffle(buffer_size=shuffle_buffer, use_rng=use_rngs),
            map_transform(
                pipeline.pre_batch_map_func(training=True), num_parallel_calls=AUTOTUNE
            ),
            batcher,
            map_transform(pipeline.post_batch_map_func(training=True)),
        ],
    )

    # shouldn't be any random calls in validation map functions - save rng creation.
    map_transform = functools.partial(transforms.map_transform, use_rng=False)
    if validation_source is not None:
        validation_source = sources.apply(
            validation_source,
            [
                map_transform(
                    pipeline.pre_cache_map_funcs(training=False),
                    num_parallel_calls=AUTOTUNE,
                ),
                map_transform(
                    pipeline.pre_batch_map_func(training=False),
                    num_parallel_calls=AUTOTUNE,
                ),
                batcher,
                map_transform(pipeline.post_batch_map_func(training=False)),
                validation_cache,
            ],
        )
    return Trainable(
        model=model,
        train_source=train_source,
        validation_source=validation_source,
        callbacks=callbacks,
        name=name,
    )
