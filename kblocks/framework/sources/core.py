import abc
from typing import Any, Callable, Mapping, Optional, Union

import gin
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from tqdm import tqdm

Split = Union[str, tfds.Split]
gin.external_configurable(tfds.ReadConfig, module="tfds")


@gin.configurable(module="kb.framework")
class DataSource(abc.ABC):
    @property
    def element_spec(self):
        return self.get_dataset("train").element_spec

    @abc.abstractmethod
    def get_dataset(self, split: Split) -> tf.data.Dataset:
        raise NotImplementedError("Abstract method")

    def epoch_length(self, split: Split) -> Optional[int]:
        cardinality = tf.data.experimental.cardinality(self.get_dataset(split)).numpy()
        return None if cardinality < 0 else cardinality

    @property
    def meta(self) -> Mapping[str, Any]:
        return {}


class DelegatingSource(DataSource):
    """
    DataSource implementation that delegates implementations to a base DataSource.

    While this class can be instantiated directly, it's primary purpose is as a base
    class.
    """

    def __init__(self, base: DataSource):
        self._base = base

    def get_dataset(self, split: Split) -> tf.data.Dataset:
        return self._base.get_dataset(split)

    def epoch_length(self, split: Split) -> int:
        return self._base.epoch_length(split)

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._base.meta

    @property
    def element_spec(self):
        return self._base.element_spec


@gin.configurable(module="kb.framework")
class BaseSource(DataSource):
    def __init__(
        self,
        dataset_fn: Callable[[Split], tf.data.Dataset],
        epoch_lengths: Optional[Mapping[Split, int]],
        meta: Optional[Mapping[str, Any]] = None,
    ):
        self._dataset_fn = dataset_fn
        self._epoch_lengths = epoch_lengths or {}
        self._meta = meta

    @property
    def meta(self) -> Mapping[str, Any]:
        return {} if self._meta is None else self._meta

    def get_dataset(self, split: Split) -> tf.data.Dataset:
        return self._dataset_fn(split)

    def epoch_length(self, split: Split) -> Optional[int]:
        length = self._epoch_lengths.get(split)
        if length is None:
            length = tf.data.experimental.cardinality(self.get_dataset(split)).numpy()
            self._epoch_lengths[split] = length
        return None if length < 0 else length


@gin.configurable(module="kb.framework")
class TfdsSource(DataSource):
    def __init__(
        self,
        builder: Union[str, tfds.core.DatasetBuilder],
        as_supervised: bool = True,
        download_and_prepare: bool = True,
        split_map=None,
        shuffle_files: Optional[bool] = None,
        read_config: Optional[tfds.ReadConfig] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ):
        self._shuffle_files = shuffle_files
        if isinstance(builder, str):
            builder = tfds.builder(builder)
        assert isinstance(builder, tfds.core.DatasetBuilder)
        self.builder: tfds.core.DatasetBuilder = builder
        if download_and_prepare:
            self.builder.download_and_prepare()
        self.as_supervised = as_supervised

        self._split_map = {} if split_map is None else split_map
        if meta is None:
            # check for classification
            meta = {}
            if as_supervised:
                info = self.builder.info
                label = info.features[info.supervised_keys[1]]
                if hasattr(label, "num_classes"):
                    meta = dict(num_classes=label.num_classes)
        self._meta = meta
        self._read_config = read_config

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta

    def _split(self, split):
        return self._split_map.get(split, split)

    def epoch_length(self, split: Split) -> int:
        try:
            return self.builder.info.splits[self._split(split)].num_examples
        except KeyError:
            return super().epoch_length(split)

    def get_dataset(self, split) -> tf.data.Dataset:
        mapped_split = self._split(split)
        if self._shuffle_files is None:
            shuffle_files = split == "train"
        else:
            shuffle_files = self._shuffle_files
        dataset = self.builder.as_dataset(
            split=mapped_split,
            as_supervised=self.as_supervised,
            shuffle_files=shuffle_files,
            read_config=self._read_config,
        )
        return dataset


@gin.configurable(module="kb.framework")
def run_data_source(
    source: DataSource, callback: Optional[Callable] = None, split: Split = "train"
):
    if callback is None:
        # run basic benchmark
        logging.info("No callback provided - running for basic timings")
        for example in tqdm(source.get_dataset(split), total=source.epoch_length):
            pass
    else:
        for example in source.get_dataset(split):
            callback(example)