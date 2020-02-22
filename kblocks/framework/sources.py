from typing import Union, Callable, Optional, Dict, Any
import abc
from absl import logging
from copy import deepcopy
import gin
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

from .pipelines import DataPipeline

Split = Union[str, tfds.Split]
gin.external_configurable(tfds.ReadConfig, module='tfds')


@gin.configurable(module='kb.framework')
class DataSource(abc.ABC):

    @property
    def example_spec(self):
        return self.get_dataset('train').element_spec

    @abc.abstractmethod
    def get_dataset(self, split: Split):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def examples_per_epoch(self, split: Split):
        raise NotImplementedError('Abstract method')

    @property
    def meta(self) -> Dict[str, Any]:
        return {}


@gin.configurable(module='kb.framework')
class BaseSource(DataSource):

    def __init__(self, dataset_fn: Callable[[Split], tf.data.Dataset],
                 examples_per_epoch, meta):
        self._dataset_fn = dataset_fn
        self._examples_per_epoch = examples_per_epoch
        self._meta = meta

    @property
    def meta(self):
        return deepcopy(self._meta)

    def get_dataset(self, split: Split):
        return self._dataset_fn(split)

    def examples_per_epoch(self, split: Split):
        return self._examples_per_epoch[split]


@gin.configurable(module='kb.framework')
class TfdsSource(DataSource):

    def __init__(self,
                 builder: Union[str, tfds.core.DatasetBuilder],
                 as_supervised: bool = True,
                 download_and_prepare: bool = True,
                 split_map=None,
                 examples_per_epoch=None,
                 shuffle_files: Optional[bool] = None,
                 read_config: Optional[tfds.ReadConfig] = None,
                 meta=None):
        self._shuffle_files = shuffle_files
        self._examples_per_epoch = examples_per_epoch
        if isinstance(builder, str):
            builder = tfds.builder(builder)
        assert (isinstance(builder, tfds.core.DatasetBuilder))
        self.builder: tfds.core.DatasetBuilder = builder
        if download_and_prepare:
            self.builder.download_and_prepare()
        self.as_supervised = as_supervised

        if split_map is None:
            split_map = {}
        self._split_map = split_map
        if meta is None:
            # check for classification
            meta = {}
            if as_supervised:
                info = self.builder.info
                label = info.features[info.supervised_keys[1]]
                if hasattr(label, 'num_classes'):
                    meta = dict(num_classes=label.num_classes)
        self._meta = meta
        self._read_config = read_config

    @property
    def meta(self) -> Dict[str, Any]:
        return deepcopy(self._meta)

    def _split(self, split):
        return self._split_map.get(split, split)

    def examples_per_epoch(self, split):
        if self._examples_per_epoch is None:
            split = self._split(split)
            return self.builder.info.splits[split].num_examples
        else:
            if isinstance(self._examples_per_epoch, int):
                return self._examples_per_epoch
            else:
                return self._examples_per_epoch[split]

    def get_dataset(self, split):
        mapped_split = self._split(split)
        if isinstance(split, dict):
            RI = tfds.core.tfrecords_reader.ReadInstruction
            ri = None
            for k, (from_, to) in mapped_split.items():
                nex = RI(k, from_=from_ * 100, to=to * 100, unit='%')
                if ri is None:
                    ri = nex
                else:
                    ri = ri + nex
            mapped_split = ri

        shuffle_files = self._shuffle_files
        if shuffle_files is None:
            shuffle_files = split == 'train'
        dataset = self.builder.as_dataset(split=mapped_split,
                                          as_supervised=self.as_supervised,
                                          shuffle_files=shuffle_files,
                                          read_config=self._read_config)
        if self._examples_per_epoch is not None:
            dataset = dataset.take(self.examples_per_epoch(split))
        return dataset


@gin.configurable(module='kb.framework')
class PipelinedSource(DataSource):

    def __init__(self, source: DataSource, pipeline: DataPipeline, meta=None):
        assert (isinstance(source, DataSource))
        assert (isinstance(pipeline, DataPipeline))
        self._base_source = source
        self._pipeline = pipeline
        if meta is None:
            meta = source.meta
        self._meta = meta

    @property
    def meta(self) -> Dict[str, Any]:
        return deepcopy(self._meta)

    @property
    def base_source(self):
        return self._base_source

    @property
    def pipeline(self):
        return self._pipeline

    def get_dataset(self, split: str):
        return self._pipeline(self._base_source.get_dataset(split), split)

    def examples_per_epoch(self, split: str):
        return self._base_source.examples_per_epoch(
            split) // self._pipeline.batch_size

    @property
    def example_spec(self):
        with tf.keras.backend.learning_phase_scope(False):
            return self._pipeline.processed_spec(self._base_source.example_spec)


@gin.configurable(module='kb.framework')
def run_data_source(source: DataSource,
                    callback: Optional[Callable] = None,
                    split: Split = 'train'):
    if callback is None:
        # run basic benchmark
        logging.info('No callback provided - running for basic timings')
        for example in tqdm(source.get_dataset(split),
                            total=source.examples_per_epoch):
            pass
    else:
        for example in source.get_dataset(split):
            callback(example)
