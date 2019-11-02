from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gin
import tensorflow as tf
from kblocks.spec import to_spec, specs_are_consistent
from kblocks.layers import Input
from kblocks.tf_typing import NestedTensorLike
from kblocks.tf_typing import NestedTensorLikeSpec
from typing import Callable, List


class Pipeline(abc.ABC):
    """
    Abstract base class for feature pipelines.

    Derived classes must implement:
        - _build

    Derived classes may also which to override:
        - pre_batch_map (defaults to identity)
        - post_batch_map (defaults to identity)
    """

    def pre_batch_map(self, args: NestedTensorLike) -> NestedTensorLike:
        """Mapping applied to dataset features before batching."""
        return args

    def post_batch_map(self, args: NestedTensorLike) -> NestedTensorLike:
        """Mapping applied to dataset features after batching."""
        return args

    @abc.abstractproperty
    def features_spec(self) -> NestedTensorLikeSpec:
        """Spec for pre-prebatch features."""
        raise NotImplementedError

    @property
    def outputs_spec(self) -> NestedTensorLikeSpec:
        """Spec of model outputs."""
        return tf.nest.map_structure(to_spec, self.model.outputs)

    @abc.abstractproperty
    def model(self) -> tf.keras.Model:
        """`tf.keras.Model`."""
        raise NotImplementedError


@gin.configurable(module='kb.framework',
                  blacklist=['features_spec', 'outputs_spec'])
class ModelPipeline(Pipeline):

    def __init__(
            self, features_spec: NestedTensorLikeSpec,
            outputs_spec: NestedTensorLikeSpec,
            model_fn: Callable[[NestedTensorLike, NestedTensorLikeSpec], tf.
                               keras.Model]):
        if model_fn is None:
            raise ValueError('`model_fn` cannot be `None`')
        self._features_spec = features_spec
        inputs = tf.nest.map_structure(
            lambda s: Input(shape=s.shape,
                            dtype=s.dtype,
                            ragged=isinstance(s, tf.RaggedTensorSpec)),
            features_spec)

        self._model = model_fn(inputs, outputs_spec)
        self._outputs_spec = outputs_spec
        actual_outputs_spec = tf.nest.map_structure(to_spec, self.model.outputs)

        expected_specs: List[tf.TensorSpec] = tf.nest.flatten(outputs_spec)
        actual_specs: List[tf.TensorSpec] = tf.nest.flatten(actual_outputs_spec)
        for actual, expected in zip(actual_specs, expected_specs):
            if not specs_are_consistent(actual, expected):
                raise ValueError(
                    'model_fn returned a model with outputs inconsistent with '
                    'outputs_spec. Requires {}, got {}'.format(
                        outputs_spec, actual_outputs_spec))

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def features_spec(self):
        return self._features_spec

    @property
    def outputs_spec(self):
        return self._outputs_spec
