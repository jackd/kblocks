from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Union
import tensorflow as tf
import tensorflow_datasets as tfds
from kblocks.framework.problems.core import Problem, Split
import six
import gin
from typing import Optional, Callable, Union, Mapping


@gin.configurable(module='kb.framework')
class TfdsProblem(Problem):

    def __init__(self,
                 builder: Union[str, tfds.core.DatasetBuilder],
                 loss,
                 metrics=(),
                 objective=None,
                 features_spec=None,
                 outputs_spec=None,
                 as_supervised=True,
                 split_map: Optional[Mapping[Split, Split]] = None,
                 download_and_prepare=True,
                 pre_batch_map: Optional[
                     Union[Callable, Mapping[Split, Callable]]] = None,
                 shuffle_buffer: int = 512):
        if isinstance(builder, six.string_types):
            builder = tfds.builder(builder)
        assert (isinstance(builder, tfds.core.DatasetBuilder))
        self.builder: tfds.core.DatasetBuilder = builder
        if download_and_prepare:
            self.builder.download_and_prepare()
        self.as_supervised = as_supervised

        if split_map is None:
            split_map = {}
        self._split_map = split_map
        self._pre_batch_map = pre_batch_map
        super(TfdsProblem, self).__init__(loss=loss,
                                          metrics=metrics,
                                          objective=objective,
                                          features_spec=features_spec,
                                          outputs_spec=outputs_spec,
                                          shuffle_buffer=shuffle_buffer)

    def _split(self, split):
        return self._split_map.get(split, split)

    def _examples_per_epoch(self, split):
        split = self._split(split)
        return self.builder.info.splits[split].num_examples

    def _get_base_dataset(self, split):
        split = self._split(split)
        if isinstance(split, dict):
            RI = tfds.core.tfrecords_reader.ReadInstruction
            ri = None
            for k, (from_, to) in split.items():
                nex = RI(k, from_=from_ * 100, to=to * 100, unit='%')
                if ri is None:
                    ri = nex
                else:
                    ri = ri + nex
            split = ri

        dataset = self.builder.as_dataset(split=split,
                                          as_supervised=self.as_supervised)
        if self._pre_batch_map is not None:
            if isinstance(self._pre_batch_map, Mapping):
                map_fn = self._pre_batch_map[split]
            elif isinstance(self._pre_batch_map, Callable):
                map_fn = self._pre_batch_map
            else:
                raise RuntimeError(
                    'Unexpected type of self._pre_batch_map {}'.format(
                        type(self._pre_batch_map)))
            dataset = dataset.map(map_fn, tf.data.experimental.AUTOTUNE)
        return dataset
