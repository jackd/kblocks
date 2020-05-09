"""
Simple re-implementation of tf.data.Dataset.cache.

I swear there's a memory leak in the official implementation.
"""
import functools
import os

import gin
import tensorflow as tf
from absl import logging
from tqdm import tqdm

from kblocks.extras.cache import core

AUTOTUNE = tf.data.experimental.AUTOTUNE


def serialize_example(*args, **kwargs):
    updater = kwargs.pop("updater")
    flat_example = tf.nest.flatten((args, kwargs), expand_composites=True)
    flat_strings = tuple(tf.io.serialize_tensor(x) for x in flat_example)
    tf.py_function(updater, [], [])
    flat_strings = tf.stack(flat_strings)
    return tf.io.serialize_tensor(flat_strings)


def deserializer(spec):
    flat_spec = tf.nest.flatten(spec, expand_composites=True)

    def deserialize_example(example):
        flat_strings = tf.io.parse_tensor(example, tf.string)
        unstacked = tf.unstack(flat_strings, len(flat_spec))
        flat_example = tuple(
            tf.io.parse_tensor(x, s.dtype) for x, s in zip(unstacked, flat_spec)
        )
        for example, s in zip(flat_example, flat_spec):
            example.set_shape(s.shape)
        return tf.nest.pack_sequence_as(spec, flat_example, expand_composites=True)

    return deserialize_example


def _write_dataset(dataset: tf.data.Dataset, path):
    if os.path.isfile(path):
        logging.info(f"Reusing data found at {path}")
        return
    logging.info(f"Writing data to {path}")
    writer = tf.data.experimental.TFRecordWriter(path)
    try:
        return writer.write(dataset)
    except (Exception, KeyboardInterrupt):
        if os.path.isfile(path):
            os.remove(path)
        logging.info(
            "Error writing tfrecords. " f"Removing partially written file at {path}"
        )
        raise


def _cache_dataset(
    dataset: tf.data.Dataset, cache_dir: str, num_parallel_calls=AUTOTUNE,
):

    spec = dataset.element_spec
    path = os.path.join(cache_dir, "serialized.tfrecords")
    if not os.path.isfile(path):
        updater = tqdm()
        write_op = _write_dataset(
            dataset.map(
                functools.partial(serialize_example, updater=updater.update),
                num_parallel_calls,
            ),
            path,
        )
        if not tf.executing_eagerly():
            with tf.compat.v1.Session() as sess:
                sess.run(write_op)
        updater.close()
    return tf.data.TFRecordDataset(path,).map(deserializer(spec), num_parallel_calls)


@gin.configurable(module="kb.cache")
class TFRecordsCacheManager(core.BaseCacheManager):
    def __init__(
        self, cache_dir: str, num_parallel_calls: int = AUTOTUNE,
    ):
        super().__init__(cache_dir=cache_dir)
        self._num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        return _cache_dataset(
            dataset, self.cache_dir, num_parallel_calls=self._num_parallel_calls,
        )
