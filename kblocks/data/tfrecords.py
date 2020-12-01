"""
Simple re-implementation of tf.data.Dataset.cache.

I feel there's a memory leak -somewhere- related to the official implementation.
"""
from typing import Optional

import tensorflow as tf
from absl import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE


def serialize_example(*args, **kwargs):
    flat_example = tf.nest.flatten((args, kwargs), expand_composites=True)
    flat_strings = tuple(tf.io.serialize_tensor(x) for x in flat_example)
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
        for el, s in zip(flat_example, flat_spec):
            el.set_shape(s.shape)
        return tf.nest.pack_sequence_as(spec, flat_example, expand_composites=True)

    return deserialize_example


def save_serialized(
    serialized: tf.data.Dataset, path: str, compression: Optional[str] = None
):
    writer = tf.data.experimental.TFRecordWriter(path, compression_type=compression)
    return writer.write(serialized)


def save(dataset: tf.data.Dataset, path: str, compression: Optional[str] = None):
    serialized = dataset.map(
        serialize_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # data corruptions with num_parallel_calls != 1?
    # https://github.com/tensorflow/tensorflow/issues/13463
    return save_serialized(serialized, path=path, compression=compression)


def load(path: str, compression: Optional[str] = None):
    return tf.data.TFRecordDataset(path, compression_type=compression)


def parse(loaded: tf.data.Dataset, spec, num_parallel_calls=1, deterministic=None):
    return loaded.map(
        deserializer(spec), num_parallel_calls, deterministic=deterministic
    )


def tfrecords_cache(
    dataset: tf.data.Dataset,
    cache_dir: str,
    num_parallel_calls: int = 1,
    compression: Optional[str] = None,
    deterministic: Optional[bool] = None,
):
    if tf.executing_eagerly():
        tf.io.gfile.makedirs(cache_dir)

    cardinality = dataset.cardinality()
    path = cache_dir + "/serialized.tfrecords"  # must work in graph mode
    if tf.shape(tf.io.matching_files(path))[0] == 0:
        logging.info(f"Saving tfrecords dataset to {path}")
        save(dataset, path, compression=compression)
    return parse(
        load(path, compression=compression),
        spec=dataset.element_spec,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    ).apply(tf.data.experimental.assert_cardinality(cardinality))
