"""
Simple re-implementation of tf.data.Dataset.cache.

I swear there's a memory leak in the official implementation.
"""
from typing import Optional
import os
from absl import logging
from tqdm import tqdm
import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_datasets.core import example_serializer
from . import core


def _pack_as_dict(element_spec):
    return {
        f'features-{i:04d}': s for i, s in enumerate(
            tf.nest.flatten(element_spec, expand_composites=True))
    }


def _pack_as_original(element_spec, data):
    if element_spec is None:
        return data
    return tf.nest.pack_sequence_as(element_spec, tf.nest.flatten(data))


def _write_dataset(dataset, serializer, paths, options=None):
    num_shards = len(paths)
    for path in paths:
        if os.path.isfile(path):
            raise ValueError(f'file already exists at {path}')
    writers = [tf.io.TFRecordWriter(path, options) for path in paths]
    try:
        logging.info('Caching datasets')
        for i, example in tqdm(enumerate(core.dataset_iterator(dataset)),
                               desc=f'Caching dataset'):
            record = serializer.serialize_example(example)
            writers[i % num_shards].write(record)

    except (Exception, KeyboardInterrupt):
        for writer in writers:
            writer.close()
        for path in paths:
            if os.path.isfile(path):
                os.remove(path)
        raise
    for writer in writers:
        writer.close()


def _cache_dataset(dataset: tf.data.Dataset,
                   cache_dir: str,
                   num_shards: int,
                   num_parallel_reads=None):

    spec = dataset.element_spec
    if isinstance(spec, dict):
        base_spec = None
    else:
        base_spec = spec
        spec = _pack_as_dict(spec)
        dataset = dataset.map(lambda *args: _pack_as_dict(args))
    features = tf.nest.map_structure(
        lambda s: tfds.core.features.feature.Tensor(shape=s.shape.as_list(),
                                                    dtype=s.dtype),
        spec,
        expand_composites=True)
    features = tfds.core.features.features_dict.FeaturesDict(features)
    features._set_top_level()
    example_specs = features.get_serialized_info()

    serializer = example_serializer.ExampleSerializer(example_specs)
    paths = [
        os.path.join(cache_dir, f'cache-{i:04d}-{num_shards:04d}.tfrecords')
        for i in range(num_shards)
    ]
    if all(os.path.isfile(path) for path in paths):
        logging.info(f'Reusing existing cache files at {cache_dir}')
    else:
        _write_dataset(dataset, serializer, paths, options=None)

    dataset = tf.data.TFRecordDataset(paths,
                                      num_parallel_reads=num_parallel_reads)
    parser = tfds.core.example_parser.ExampleParser(example_specs)
    dataset = dataset.map(lambda x: _pack_as_original(
        base_spec, features.decode_example(parser.parse_example(x))))
    return dataset


@gin.configurable(module='kb.framework')
class TFRecordsCacheManager(core.BaseCacheManager):

    def __init__(self,
                 cache_dir: str,
                 num_shards: int,
                 num_parallel_reads: Optional[int] = None):
        if num_shards < 1:
            raise ValueError(f'num_shards must be at least 1, got {num_shards}')
        super().__init__(cache_dir=cache_dir)
        self._num_shards = num_shards
        if num_parallel_reads is None:
            num_parallel_reads = num_shards
        self._num_parallel_reads = num_parallel_reads

    def __call__(self, dataset):
        return _cache_dataset(dataset, self.cache_dir, self._num_shards,
                              self._num_parallel_reads)
