from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union
from typing import Optional
from typing import Sequence
from typing import List
from typing import Tuple
from typing import Dict
from typing import Any
from typing import Callable

import abc
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import six

from kblocks.tf_typing import NestedTensorLikeSpec
from kblocks.tf_typing import NestedTensorLike

Metric = tf.keras.metrics.Metric
Loss = tf.keras.losses.Loss

Split = Union[str, tfds.Split]


def _batches(num_examples, batch_size=None):
    if batch_size is not None:
        num_examples //= batch_size
    return num_examples


@gin.configurable(module='kb.framework')
class Objective(object):

    def __init__(self,
                 name: str,
                 split: Split = 'validation',
                 mode: str = 'max'):
        self._name = name
        self._mode = mode
        self._split = split

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def split(self):
        return self._split

    def get_config(self) -> Dict:
        return dict(name=self.name, mode=self.mode, split=self.split)

    @classmethod
    def get(cls, identifier) -> 'Objective':
        if isinstance(identifier, Objective):
            return identifier
        if isinstance(identifier, (list, tuple)):
            return Objective(*identifier)
        elif isinstance(identifier, dict):
            return Objective(**identifier)
        elif isinstance(identifier, six.string_types):
            return Objective(identifier)
        else:
            raise TypeError(
                'Cannot convert identifier {} into an Objective'.format(
                    identifier))


class Problem(abc.ABC):
    """
    Abstract base class for problems: datasets, losses and metrics etc.

    Derived classes should call this constructor and must implement:
        - _get_base_dataset
        - _examples_per_epoch

    Optionally, may override:
        - post_batch_map

    See `kblocks.framework.problems.tfds.TfdsProblem` for example.

    Args:
        loss: `tf.keras.losses.Loss` instance.
        metrics: iterable of `tf.keras.metrics.Metric` instances.
        objective: `Objective` instance. Defaults to validation version
            of first metric.
        features_spec: structure of `tf.TensorSpec` (or
            Ragged/Sparse variants) of unbatched dataset.
        outputs_spec: structure of `tf.TensorSpec` (or
            Ragged/Sparse variants) of trained model. This defaults to
            the spec of the batched/mapped labels.
        shuffle_buffer: typical value used for shuffling (though dataset
            pipelines may use any value they wish).
    """

    def __init__(
            self,
            loss: Union[Loss, Sequence[Loss]],
            metrics: Union[Sequence[Metric], Sequence[Sequence[Metric]]] = (),
            objective: Optional[Union[Objective, str]] = None,
            features_spec: Optional[NestedTensorLikeSpec] = None,
            outputs_spec: Optional[NestedTensorLikeSpec] = None,
            shuffle_buffer: int = 512):
        if isinstance(loss, Sequence):
            self._loss = tuple(tf.keras.losses.get(l) for l in loss)
        else:
            self._loss = tf.keras.losses.get(loss)
        if metrics is None or len(metrics) == 0:
            self._metrics: Tuple[Metric, ...] = ()
        elif isinstance(metrics[0], Sequence):
            self._metrics = tuple(
                tuple(tf.keras.metrics.get(mi) for mi in m) for m in metrics)
            if objective is None:
                if len(metrics) == 0 or len(metrics[0]) == 0:
                    l0 = loss[0] if isinstance(loss, Sequence) else loss
                    objective = Objective(l0.name, mode='min')
                else:
                    objective = self.metrics[0][0].name
        else:
            self._metrics: Tuple[tf.keras.metrics.Metric] = tuple(
                tf.keras.metrics.get(m) for m in metrics)
            if objective is None:
                if len(self._metrics) == 0:
                    l0 = loss[0] if isinstance(loss, Sequence) else loss
                    objective = Objective(l0.name, mode='min')
                else:
                    objective = self._metrics[0].name
        self._objective = Objective.get(objective)
        self._features_spec = features_spec
        self._outputs_spec = outputs_spec
        self._shuffle_buffer = shuffle_buffer

    _stack = []

    def __enter__(self):
        Problem._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        out = Problem._stack.pop()
        if out != self:
            raise RuntimeError('Should be popping self off stack...')

    @staticmethod
    def default():
        if len(Problem._stack) == 0:
            raise RuntimeError(
                'Cannot get default problem: at least one must be open in a '
                'context first.')
        return Problem._stack[-1]

    @property
    def shuffle_buffer(self):
        return self._shuffle_buffer

    @property
    def objective(self) -> Optional[Objective]:
        return self._objective

    @property
    def metrics(self) -> Union[List[List[Metric]], List[Metric]]:
        if isinstance(self._metrics[0], tuple):
            return list(list(m) for m in self._metrics)
        else:
            return list(self._metrics)

    @property
    def loss(self) -> Union[Loss, List[Loss]]:
        return self._loss if isinstance(self._loss, Loss) else list(self._loss)

    @property
    def features_spec(self) -> NestedTensorLikeSpec:
        """Pre-batch features spec."""
        if self._features_spec is None:
            self._features_spec = self._get_base_dataset(
                'train').element_spec[0]
        return self._features_spec

    @property
    def outputs_spec(self) -> NestedTensorLikeSpec:
        """Post-batch outputs_spec."""
        if self._outputs_spec is None:
            dataset = self._get_base_dataset('train').map(
                lambda *args: args[1:]).batch(2).map(self.post_batch_map)
            spec = dataset.element_spec
            if isinstance(spec, Sequence) and len(spec) == 2:
                # labels, weights
                spec = spec[0]
            self._outputs_spec = spec  # default: same as labels
        return self._outputs_spec

    @abc.abstractmethod
    def _examples_per_epoch(self, split: Split) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_base_dataset(self, split: Split) -> tf.data.Dataset:
        raise NotImplementedError

    def examples_per_epoch(self,
                           split: Split = tfds.Split.TRAIN,
                           batch_size: Optional[int] = None) -> int:
        return tf.nest.map_structure(
            lambda split: _batches(self._examples_per_epoch(split), batch_size),
            split)

    def get_base_dataset(self,
                         split: Split = tfds.Split.TRAIN) -> tf.data.Dataset:
        return tf.nest.map_structure(self._get_base_dataset, split)

    def post_batch_map(self,
                       labels: NestedTensorLike,
                       weights: Optional[NestedTensorLike] = None
                      ) -> NestedTensorLike:
        return labels if weights is None else (labels, weights)


@gin.configurable(module='kb.framework')
def run_problem(problem: Problem,
                callback: Callable[[NestedTensorLike, NestedTensorLike], None],
                split: Split = 'train',
                shuffle: bool = True):
    with problem:
        dataset = problem.get_base_dataset(split=split)
        if shuffle:
            dataset = dataset.shuffle(problem.shuffle_buffer)
        for example, label in dataset:
            callback(example, label)
