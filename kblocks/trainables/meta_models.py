from typing import Callable, Optional, Sequence, Tuple

import gin
import tensorflow as tf

from kblocks.data import sources, transforms
from kblocks.trainables.core import Trainable
from meta_model import pipeline as pl


def _get_map_transforms(pipeline: pl.Pipeline, training: bool):
    # shouldn't be any random calls in validation map functions - save rng creation.
    return (
        transforms.Map(
            pipeline.pre_cache_map_func(training=training),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ),
        transforms.Map(
            pipeline.pre_batch_map_func(training=training),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ),
        transforms.Map(
            pipeline.post_batch_map_func(training=training), num_parallel_calls=1
        ),
    )


def _get_sources(
    pipeline: pl.Pipeline,
    batcher: transforms.Transform,
    train_source: sources.DataSource,
    validation_source: Optional[sources.DataSource],
    shuffle_buffer: Optional[int],
    train_cache: Optional[transforms.Transform],
    validation_cache: Optional[transforms.Transform],
) -> Tuple[Optional[int], sources.DataSource, Optional[sources.DataSource]]:
    pre_cache_map, pre_batch_map, post_batch_map = _get_map_transforms(
        pipeline, training=True
    )
    shuffler = None if shuffle_buffer is None else transforms.Shuffle(shuffle_buffer)

    train_source_tmp = sources.apply(
        train_source,
        [
            pre_cache_map,
            train_cache,
            transforms.Repeat(),
            shuffler,
            pre_batch_map,
            batcher,
            post_batch_map,
        ],
    )
    # calculate steps-per-epoch in scratch-source
    scratch_source = sources.apply(
        train_source, [pre_cache_map, pre_batch_map, batcher]
    )
    steps_per_epoch = int(scratch_source.dataset.cardinality().numpy())
    train_source = train_source_tmp
    del train_source_tmp, scratch_source

    if validation_source is not None:
        pre_cache_map, pre_batch_map, post_batch_map = _get_map_transforms(
            pipeline, training=False
        )
        validation_source = sources.apply(
            validation_source,
            [
                pre_cache_map,
                pre_batch_map,
                batcher,
                post_batch_map,
                validation_cache,  # no shuffling or data augmentation, so cache last
            ],
        )
    return steps_per_epoch, train_source, validation_source


@gin.configurable(module="kb.trainable")
def build_meta_model_trainable(
    meta_model_fn: Callable,
    batcher: transforms.Transform,
    train_source: sources.DataSource,
    validation_source: Optional[sources.DataSource] = None,
    shuffle_buffer: Optional[int] = None,
    train_cache: Optional[transforms.Transform] = None,
    validation_cache: Optional[transforms.Transform] = None,
    loss: Optional[tf.keras.losses.Loss] = None,
    metrics: Optional = None,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    callbacks: Sequence[tf.keras.callbacks.Callback] = (),
    name: Optional[str] = None,
) -> Trainable:
    """
    Build a trainable using `meta_model` package.

    Args:
        TODO

    Returns:
        `Trainable` with model and data source maps defined by `meta_moodel_fn`.
    """
    pipeline, model = pl.build_pipelined_model(
        meta_model_fn, train_source.dataset.element_spec, batcher
    )
    if optimizer is None:
        assert loss is None
        assert metrics is None
    else:
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    steps_per_epoch, train_source, validation_source = _get_sources(
        pipeline=pipeline,
        batcher=batcher,
        train_source=train_source,
        validation_source=validation_source,
        shuffle_buffer=shuffle_buffer,
        train_cache=train_cache,
        validation_cache=validation_cache,
    )
    return Trainable(
        model=model,
        train_source=train_source,
        validation_source=validation_source,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        name=name,
    )
