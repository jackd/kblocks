import os
from typing import Any, Callable, Iterable, Optional

import gin
import tensorflow as tf
import tfrng
from meta_model import pipeline as pl

from kblocks.data import Transform
from kblocks.path import expand
from kblocks.trainables.core import Trainable

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _get_map_funcs(pipeline, training):
    return (
        pipeline.pre_cache_map_func(training),
        pipeline.pre_batch_map_func(training),
        pipeline.post_batch_map_func(training),
    )


def chain_map_funcs(*map_funcs):
    def chained(*args):
        for map_func in map_funcs:
            args = map_func(*args)
            if tf.is_tensor(args):
                args = (args,)
        if len(args) == 1:
            (args,) = args
        return args

    return chained


@gin.configurable(module="kb.trainables")
def build_meta_model_trainable(
    meta_model_func: Callable,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    batcher: Transform,
    shuffle_buffer: int,
    compiler: Optional[Callable[[tf.keras.Model], Any]] = None,
    cache_factory: Optional[Callable[[str], Transform]] = None,
    cache_dir: Optional[str] = None,
    cache_repeats: Optional[int] = None,
    train_augment_func: Optional[Callable] = None,
    validation_augment_func: Optional[Callable] = None,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    seed: Optional[int] = None,
):
    # this version DOES bleed examples from 1 train epoch to the next
    # via a final full batch and shuffle buffer
    # Hopefully it gets rid of the memory leak we see?
    if cache_factory is not None:
        assert cache_dir is not None
        assert cache_repeats is not None
        cache_dir = expand(cache_dir)
    else:
        assert cache_dir is None
        assert cache_repeats is None

    if validation_augment_func:
        validation_dataset = validation_dataset.map(
            validation_augment_func, num_parallel_calls=AUTOTUNE
        )
    pipeline, model = pl.build_pipelined_model(
        meta_model_func, validation_dataset.element_spec, batcher
    )
    if compiler is not None:
        compiler(model)

    # finalize validation_dataset
    pre_cache, pre_batch, post_batch = _get_map_funcs(pipeline, training=False)
    validation_dataset = (
        validation_dataset.map(chain_map_funcs(pre_cache, pre_batch), AUTOTUNE)
        .apply(batcher)
        .map(post_batch, AUTOTUNE)
    )
    if cache_factory is not None:
        validation_dataset = validation_dataset.apply(
            cache_factory(os.path.join(cache_dir, "validation"))
        )

    # train_data
    steps_per_epoch = tf.keras.backend.get_value(
        train_dataset.apply(batcher).cardinality()
    )
    pre_cache, pre_batch, post_batch = _get_map_funcs(pipeline, training=True)

    if cache_factory is None:
        train_dataset = train_dataset.repeat().shuffle(shuffle_buffer, seed=seed)
        if train_augment_func is None:
            train_dataset = (
                train_dataset.map(
                    chain_map_funcs(pre_cache, pre_batch),
                    num_parallel_calls=AUTOTUNE,
                )
                .apply(batcher)
                .map(post_batch, AUTOTUNE)
            )
        else:
            # cache_factory is None, train_augment_func is not
            train_dataset = (
                train_dataset.repeat()
                .apply(
                    tfrng.data.stateless_map(
                        chain_map_funcs(train_augment_func, pre_cache, pre_batch),
                        seed=seed,
                        num_parallel_calls=AUTOTUNE,
                    )
                )
                .apply(batcher)
                .map(post_batch, AUTOTUNE)
            )
    else:
        # cache_factory is not None
        # We create separately cached datasets and flat_map over them.
        # This allows us to reuse the same caches if we want to change cache_repeats
        assert cache_repeats < 1e4  # for unique path naming
        paths = [
            os.path.join(cache_dir, "train", f"repeat-{i:04d}")
            for i in range(cache_repeats)
        ]
        if train_augment_func is None:
            # No augmentation
            assert cache_repeats == 1
            (path,) = paths
            train_dataset = (
                train_dataset.map(pre_cache, num_parallel_calls=AUTOTUNE)
                .apply(cache_factory(path))
                .repeat()
            )
        else:

            def get_cached(epoch_seed, path):
                return train_dataset.apply(
                    tfrng.data.stateless_map(
                        chain_map_funcs(train_augment_func, pre_cache),
                        seed=epoch_seed,
                        num_parallel_calls=AUTOTUNE,
                    )
                ).apply(cache_factory(path))

            # train_dataset = (
            #     paths.apply(tfrng.data.with_seed(seed))
            #     .repeat()
            #     .flat_map(get_cached)
            # )
            # create cached datasets in eager mode.
            # For some cache_factory implementations this will mean all `cache_repeat`
            # cache files are run ahead of time.
            # This allows us to save iterators for training.
            datasets = [
                get_cached(s, p)
                for s, p in zip(tf.data.experimental.RandomDataset(seed), paths)
            ]
            train_dataset = (
                tf.data.Dataset.from_tensor_slices(datasets)
                .repeat()
                .flat_map(lambda ds: ds)
            )
        train_dataset = (
            train_dataset.shuffle(shuffle_buffer, seed=seed)
            .map(pre_batch, num_parallel_calls=AUTOTUNE)
            .apply(batcher)
            .map(post_batch, num_parallel_calls=AUTOTUNE)
            .repeat()  # HACK: because the assert_cardinality below raises on iteration
            # .apply(tf.data.experimental.assert_cardinality(
            #     tf.data.INFINITE_CARDINALITY))
        )
        # https://github.com/tensorflow/tensorflow/issues/45894

    # assert specs the same
    train_spec = train_dataset.element_spec
    val_spec = validation_dataset.element_spec
    tf.nest.assert_same_structure(train_spec, val_spec)
    flat_train = tf.nest.flatten(train_spec)
    flat_val = tf.nest.flatten(val_spec)
    for t, v in zip(flat_train, flat_val):
        assert t == v

    return Trainable(
        model=model,
        train_data=train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        callbacks=tuple(callbacks),
    )


# @gin.configurable(module="kb.trainables")
# def build_meta_model_trainable(
#     meta_model_func: Callable,
#     train_dataset: tf.data.Dataset,
#     validation_dataset: tf.data.Dataset,
#     batcher: Transform,
#     shuffle_buffer: int,
#     compiler: Optional[Callable[[tf.keras.Model], Any]] = None,
#     cache_factory: Optional[Callable[[str], Transform]] = None,
#     cache_dir: Optional[str] = None,
#     cache_repeats: Optional[int] = None,
#     train_augment_func: Optional[Callable] = None,
#     validation_augment_func: Optional[Callable] = None,
#     callbacks: Iterable[tf.keras.callbacks.Callback] = (),
#     seed: Optional[int] = None,
# ):
#     # this version does NOT bleed exampels from 1 training epoch to the next
#     # It requires `cache_repeats` separate shuffle buffers which are re-populated
#     # each epoch (potentially slow)
#     if cache_factory is not None:
#         assert cache_dir is not None
#         assert cache_repeats is not None
#         cache_dir = expand(cache_dir)
#     else:
#         assert cache_dir is None
#         assert cache_repeats is None

#     if validation_augment_func:
#         validation_dataset = validation_dataset.map(
#             validation_augment_func, num_parallel_calls=AUTOTUNE
#         )
#     pipeline, model = pl.build_pipelined_model(
#         meta_model_func, validation_dataset.element_spec, batcher
#     )
#     if compiler is not None:
#         compiler(model)

#     # finalize validation_dataset
#     pre_cache, pre_batch, post_batch = _get_map_funcs(pipeline, training=False)
#     validation_dataset = (
#         validation_dataset.map(lambda *args: pre_batch(*pre_cache(*args)), AUTOTUNE)
#         .apply(batcher)
#         .map(post_batch, AUTOTUNE)
#     )
#     if cache_factory is not None:
#         validation_dataset = validation_dataset.apply(
#             cache_factory(os.path.join(cache_dir, "validation"))
#         )

#     # train_data
#     steps_per_epoch = tf.keras.backend.get_value(
#         train_dataset.apply(batcher).cardinality()
#     )
#     pre_cache, pre_batch, post_batch = _get_map_funcs(pipeline, training=True)

# if cache_factory is None:
#     train_dataset = train_dataset.shuffle(shuffle_buffer, seed=seed)
#     if train_augment_func is None:
#         train_dataset = (
#             train_dataset.map(
#                 lambda *args: pre_batch(pre_cache(*args)),
#                 num_parallel_calls=AUTOTUNE,
#             )
#             .apply(batcher)
#             .map(post_batch, AUTOTUNE)
#             .repeat()
#         )
#     else:
#         # cache_factory is None, train_augment_func is not
#         train_dataset = (
#             tf.data.experimental.RandomDataset(seed)
#             .flat_map(
#                 lambda epoch_seed: train_dataset.apply(
#                     tfrng.data.stateless_map(
#                         lambda *args: pre_batch(
#                             *pre_cache(*train_augment_func(*args))
#                         ),
#                         seed=epoch_seed,
#                         num_parallel_calls=AUTOTUNE,
#                     )
#                 ).apply(batcher)
#             )
#             .map(post_batch, AUTOTUNE)
#         )
# else:
#     # cache_factory is not None
#     assert train_augment_func is not None
#     assert cache_repeats < 1e4  # for unique path naming
#     paths = [
#         os.path.join(cache_dir, "train", f"repeat-{i:04d}")
#         for i in range(cache_repeats)
#     ]

#     train_dataset = (
#         tf.data.Dataset.from_tensor_slices(paths)
#         .apply(tfrng.data.with_seed(seed))
#         .repeat()
#         .flat_map(
#             lambda epoch_seed, path: (
#                 train_dataset.apply(
#                     tfrng.data.stateless_map(
#                         lambda *args: pre_cache(*train_augment_func(*args)),
#                         seed=epoch_seed,
#                         num_parallel_calls=AUTOTUNE,
#                     )
#                 )
#                 .apply(cache_factory(path))
#                 .shuffle(shuffle_buffer, seed=epoch_seed)
#                 .map(pre_batch, AUTOTUNE)
#                 .apply(batcher)
#             )
#         )
#         .map(post_batch, AUTOTUNE)
#     )
#     # train_dataset = train_dataset.apply(
#     #     tf.data.experimental.assert_cardinality(tf.data.INFINITE_CARDINALITY)
#     # )
#     train_dataset = train_dataset.repeat()  # HACK: cannot set infinte cardinality

# # assert specs the same
# train_spec = train_dataset.element_spec
# val_spec = validation_dataset.element_spec
# tf.nest.assert_same_structure(train_spec, val_spec)
# flat_train = tf.nest.flatten(train_spec)
# flat_val = tf.nest.flatten(val_spec)
# for t, v in zip(flat_train, flat_val):
#     assert t == v

# return Trainable(
#     model=model,
#     train_data=RepeatedData(train_dataset, steps_per_epoch),
#     validation_data=RepeatedData(validation_dataset),
#     callbacks=tuple(callbacks),
# )
