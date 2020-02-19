raise NotImplementedError('deprecated')
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from typing import Any
# from typing import Callable
# from typing import Optional
# from typing import Union

# import gin
# import tensorflow as tf
# import tensorflow_datasets as tfds

# from kblocks.tf_typing import NestedTensorLikeSpec
# from kblocks.tf_typing import NestedTensorLike
# from kblocks.utils import memoized_property
# from .pipelines.core import DataPipeline
# from .source import DataSource

# Metric = tf.keras.metrics.Metric
# Loss = tf.keras.losses.Loss

# Split = Union[str, tfds.Split]

# @gin.configurable(module='kb.framework')
# class Problem(object):
#     """
#     Abstract base class for problems: datasets, losses and metrics etc.

#     Derived classes should call this constructor and must implement:
#         - get_base_dataset
#         - examples_per_epoch

#     See `kblocks.framework.problems.tfds.TfdsProblem` for example.

#     Args:
#         pipeline: dict of mapping splits to `DataPipeline`s.
#         loss: `tf.keras.losses.Loss` instance.
#         metrics: iterable of `tf.keras.metrics.Metric` instances.
#         objective: `Objective` instance. Defaults to validation version
#             of first metric.
#         outputs_spec: tf.TensorSpec (or ragged/sparse equivalent) or nested
#             structure corresponding to model outputs.
#     """

#     def __init__(self,
#                  source: DataSource,
#                  pipeline: DataPipeline,
#                  outputs_spec: Optional[NestedTensorLikeSpec] = None):
#         if isinstance(pipeline, dict):
#             assert ('train' in pipeline)
#             assert ('validation' in pipeline)
#         else:
#             assert (isinstance(pipeline, DataPipeline))
#         self._source = source
#         self._outputs_spec = outputs_spec
#         self._pipeline = pipeline

#     @property
#     def pipeline(self) -> DataPipeline:
#         return self._pipeline

#     @property
#     def source(self) -> DataSource:
#         return self._source

#     @memoized_property
#     def processed_spec(self):
#         with tf.keras.backend.learning_phase_scope(False):
#             return self.pipeline.processed_spec(self.source.example_spec)

#     @property
#     def outputs_spec(self) -> NestedTensorLikeSpec:
#         if self._outputs_spec is None:
#             # default to same as labels
#             return self.processed_spec[1]
#         return self._outputs_spec

#     @property
#     def features_spec(self) -> NestedTensorLikeSpec:
#         return self.processed_spec[0]

#     def examples_per_epoch(self, split: Split) -> int:
#         """Get the number of examples per epoch in the given split."""
#         return self._source.examples_per_epoch(split)

#     def steps_per_epoch(self, split: Split) -> int:
#         return self.examples_per_epoch(split) // self.pipeline.batch_size

#     def get_processed_dataset(self, split: Split,
#                               spec_only=False) -> tf.data.Dataset:
#         dataset = self.source.get_dataset(split)
#         with tf.keras.backend.learning_phase_scope(split == 'train'):
#             return self.pipeline(dataset)

# @gin.configurable(module='kb.framework')
# def run_problem(problem: Problem,
#                 callback: Callable[[NestedTensorLike, NestedTensorLike], None],
#                 split: Split = 'train',
#                 shuffle: bool = True):
#     dataset = problem.get_processed_dataset(split=split)
#     for args in dataset:
#         callback(*args)
