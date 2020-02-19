from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from typing import Optional, Callable
import gin

import tensorflow as tf
from tensorflow.python.keras.engine import base_layer_utils  # pylint: disable=no-name-in-module,import-error

from kblocks.tf_typing import NestedTensorLikeSpec
from kblocks.framework.problems import scope as problem_scope
from kblocks.framework.trainable import Trainable
from kblocks.framework.problems.pipelines.builder.model_builder import UnbatchedModelBuilder
from kblocks.spec import to_spec
from kblocks.extras.layers import ragged as ragged_layers
from kblocks.scope import Scope
from kblocks.tensor_dict import TensorDict
from kblocks.framework.problems.pipelines.builder.model_builder import ModelBuilder
from kblocks.framework.problems.pipelines.builder.model_builder import UnbatchedModelBuilder

from kblocks.framework.problems.pipelines.core import DataPipeline

from kblocks.tf_typing import TensorLike
from kblocks.tf_typing import TensorLikeSpec
from typing import Optional, Tuple
Lambda = tf.keras.layers.Lambda
Input = tf.keras.layers.Input

Model = tf.keras.Model
AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataPartitions(object):
    FEATURES = 0
    LABELS = 1
    WEIGHTS = 2

    @classmethod
    def all(cls):
        return (DataPartitions.FEATURES, DataPartitions.LABELS,
                DataPartitions.WEIGHTS)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise KeyError('Invalid {} {}'.format(cls, key))


def partition(data, partitions):
    if isinstance(data, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
        data = data,
    if (len(data) != len(partitions)):
        raise ValueError(
            'len(data) and len(partitions) must be equal, but {} != {}'.format(
                len(data), len(partitions)))
    num_partitions = max(partitions) + 1
    out = [[] for _ in range(num_partitions)]
    for d, p in zip(data, partitions):
        out[p].append(d)
    return out


def flatten_partition(data, partitions):
    TensorLike = (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)
    if isinstance(data, TensorLike):
        data = data,
        total_data = 1
    else:
        total_data = sum(
            (1 if isinstance(d, TensorLike) else len(d) for d in data))
    if total_data != len(partitions):
        raise ValueError(
            'total_data and len(partitions) must be equal, but {} != {}'.format(
                total_data, len(partitions)))
    out = []
    counts = [0 for _ in data]
    for p in partitions:
        out.append(data[p][counts[p]])
        counts[p] += 1
    return out


def _assert_untrainable(model: Optional[Model], name: str):
    if model is not None and model.trainable_weights:
        raise ValueError(
            '{} should not have any trainable weights'.format(name))


def _pre_batch_map(features,
                   labels=None,
                   weights=None,
                   model=None,
                   input_partitions=None,
                   output_partitions=None):
    inputs = (x for x in (features, labels, weights) if x is not None)
    if input_partitions is None:
        inputs = tf.nest.flatten(tuple(inputs))
    else:
        inputs = tuple(tf.nest.flatten(x) for x in inputs)
        inputs = flatten_partition(inputs, input_partitions)

    out = UnbatchedModelBuilder.apply(model, inputs)
    if output_partitions is None:
        if isinstance(out, list):
            out = tuple(out)
        return out
    else:
        out = partition(out, output_partitions)
        return tuple(tuple(o) for o in out)


def _post_batch_map(*inputs, model=None, output_partitions=None):
    out = ModelBuilder.apply(model, inputs)
    if output_partitions is not None:
        out = partition(out, output_partitions)
        out = tuple(tuple(o) for o in out)
    return out


def _build_model(builder: Optional[ModelBuilder], name):
    return None if builder is None else builder.build(name)


class BuiltPipeline(DataPipeline):

    def __init__(self,
                 batch_size: int,
                 pre_cache_model: Optional[Model],
                 pre_batch_model: Optional[Model],
                 post_batch_model: Model,
                 base_partitions: Optional[Tuple[int, ...]],
                 output_partitions: Tuple[int, ...],
                 cache_path: str = '',
                 shuffle_buffer: Optional[int] = None,
                 repeats: Optional[int] = None,
                 prefetch_buffer: int = AUTOTUNE,
                 num_parallel_calls: int = AUTOTUNE):
        _assert_untrainable(pre_cache_model, 'pre_cache_model')
        _assert_untrainable(pre_batch_model, 'pre_batch_model')

        if pre_batch_model is not None and pre_batch_model.trainable_weights:
            raise ValueError(
                'pre_batch_model should not have any trainable weights')
        if post_batch_model.trainable_weights:
            raise ValueError(
                'post_batch_model should not have any trainable weights')
        self._batch_size = batch_size
        self._pre_cache_model = pre_cache_model
        self._pre_batch_model = pre_batch_model
        self._post_batch_model = post_batch_model
        self._cache_path = cache_path
        self._shuffle_buffer = shuffle_buffer
        self._repeats = repeats
        self._prefetch_buffer = prefetch_buffer
        self._num_parallel_calls = num_parallel_calls
        self._base_partitions = base_partitions
        self._output_partitions = output_partitions

    def get_config(self):
        return dict(
            batch_size=self._batch_size,
            pre_cache_model=self._pre_cache_model,
            pre_batch_model=self._pre_batch_model,
            post_batch_model=self._post_batch_model,
            base_partitions=self._base_partitions,
            output_partitions=self._output_partitions,
            cache_path=self._cache_path,
            shuffle_buffer=self._shuffle_buffer,
            repeats=self._repeats,
            prefetch_buffer=self._prefetch_buffer,
            num_parallel_calls=self._num_parallel_calls,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        input_partitions = self._base_partitions
        if self._pre_cache_model is not None:
            dataset = dataset.map(
                functools.partial(_pre_batch_map,
                                  model=self._pre_cache_model,
                                  input_partitions=input_partitions),
                self._num_parallel_calls)
            input_partitions = None
            dataset = dataset.cache(self._cache_path)

        if self._repeats != -1:
            dataset = dataset.repeat(self._repeats)
        if self._shuffle_buffer is not None:
            dataset = dataset.shuffle(self._shuffle_buffer)

        if self._pre_batch_model is not None:
            output_partitions = (None if self._post_batch_model is None else
                                 self._output_partitions)
            dataset = dataset.map(
                functools.partial(_pre_batch_map,
                                  model=self._pre_batch_model,
                                  input_partitions=input_partitions,
                                  output_partitions=output_partitions),
                self._num_parallel_calls,
            )

        dataset = dataset.batch(self.batch_size)

        if self._post_batch_model is not None:
            dataset = dataset.map(
                functools.partial(_post_batch_map,
                                  model=self._post_batch_model,
                                  output_partitions=self._output_partitions),
                self._num_parallel_calls)

        if self._prefetch_buffer is not None:
            dataset = dataset.prefetch(self._prefetch_buffer)
        return dataset


class PipelineModels(object):
    PRE_CACHE = 'pre_cache'
    PRE_BATCH = 'pre_batch'
    POST_BATCH = 'post_batch'
    TRAINED = 'trained'

    @classmethod
    def validate(cls, id_: str):
        if id_ not in cls.all():
            raise ValueError('invalid PipelineModel key {}'.format(id_))

    @classmethod
    def all(cls):
        return (PipelineModels.PRE_CACHE, PipelineModels.PRE_BATCH,
                PipelineModels.POST_BATCH, PipelineModels.TRAINED)


def batch_spec(spec: TensorLikeSpec, batch_size):
    if isinstance(spec, tf.RaggedTensorSpec):
        return tf.RaggedTensorSpec(shape=(batch_size,) + spec._shape,
                                   dtype=spec._dtype)
    else:
        assert (isinstance(spec, (tf.TensorSpec, tf.SparseTensorSpec)))
        return spec.__class__(shape=(batch_size,) + spec.shape,
                              dtype=spec.dtype)


class PipelineBuilder(object):

    def __init__(self, batch_size: int, use_cache=False):
        self._batch_size = batch_size
        self._use_cache = use_cache

        if use_cache:
            self._pre_cache_builder = UnbatchedModelBuilder()
        else:
            self._pre_cache_builder = None
        self._base_partitions = []
        self._output_partitions = []
        self._pre_batch_builder = UnbatchedModelBuilder()
        self._post_batch_builder = ModelBuilder()
        self._trained_builder = ModelBuilder()
        self._builders = {
            PipelineModels.PRE_CACHE: self._pre_cache_builder,
            PipelineModels.PRE_BATCH: self._pre_batch_builder,
            PipelineModels.POST_BATCH: self._post_batch_builder,
            PipelineModels.TRAINED: self._trained_builder,
        }
        self._marks = Marks()
        self._batch_size = batch_size

    def _propagate(self, labels: TensorLike, partition: int):
        mark = self.get_mark(labels)
        if mark == PipelineModels.PRE_CACHE:
            labels = self.cache(labels)
            mark = self.get_mark(labels)
        if mark == PipelineModels.PRE_BATCH:
            labels = self.batch(labels)
            mark = self.get_mark(labels)
        if mark == PipelineModels.POST_BATCH:
            self.trained_input(labels, partition=DataPartitions.LABELS)

    def propagate_labels(self, labels: TensorLike):
        self._propagate(labels, DataPartitions.LABELS)

    def propagate_weights(self, weights: TensorLike):
        if weights is not None:
            self._propagate(weights, DataPartitions.WEIGHTS)

    @property
    def use_cache(self):
        return self._use_cache

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_mark(self, end: TensorLike) -> Optional[str]:
        return self._marks.propagate(end)

    def check_mark(self, tensor: TensorLike, mark: str,
                   name: str = 'tensor') -> None:
        actual = self.get_mark(tensor)
        if actual != mark:
            raise RuntimeError(
                'Expected {} to have mark {}, but {} has mark {}'.format(
                    name, mark, tensor, actual))

    def _pre_cache_input(self, spec: TensorLikeSpec,
                         name: Optional[str] = None) -> tf.Tensor:
        inp = self._pre_cache_builder.add_input(spec, name=name)
        self._marks[inp] = PipelineModels.PRE_CACHE
        return inp

    def cache(self,
              tensor: TensorLike,
              name: Optional[str] = None,
              partition: int = DataPartitions.FEATURES):
        if not self.use_cache:
            raise RuntimeError(
                'Cannot cache input - PipelineBuilder does not use cache')
        DataPartitions.validate(partition)
        self._marks[tensor] = PipelineModels.PRE_CACHE
        self._pre_cache_builder.add_output(tensor)
        return self._pre_batch_input(spec=to_spec(tensor), name=name)

    def _pre_batch_input(self, spec: TensorLikeSpec,
                         name: Optional[str] = None) -> tf.Tensor:
        inp = self._pre_batch_builder.add_input(spec, name=name)
        self._marks[inp] = PipelineModels.PRE_BATCH
        return inp

    def base_dataset_inputs(self, features_spec, labels_spec,
                            weights_spec=None):
        out = []
        for specs, partition in (
            (features_spec, DataPartitions.FEATURES),
            (labels_spec, DataPartitions.FEATURES),
            (weights_spec, DataPartitions.FEATURES),
        ):
            if specs is not None:
                out.append(
                    tf.nest.map_structure(
                        lambda s: self.base_input(s, partition=partition),
                        specs))
        return tuple(out)

    def base_input(self,
                   spec: TensorLikeSpec,
                   name: Optional[str] = None,
                   partition: int = DataPartitions.FEATURES):
        DataPartitions.validate(partition)
        self._base_partitions.append(partition)
        if self.use_cache:
            return self._pre_cache_input(spec, name)
        else:
            return self._pre_batch_input(spec, name)

    def _batch(self,
               tensor: TensorLike,
               ragged: Optional[bool] = None,
               name: Optional[str] = None) -> TensorLike:

        def batch(tensor: TensorLike, name: Optional[str] = None):
            self._marks[tensor] = PipelineModels.PRE_BATCH
            self._pre_batch_builder.add_output(tensor)
            spec = to_spec(tensor)
            spec = batch_spec(spec, self.batch_size)
            out = self._post_batch_builder.add_input(spec)
            self._marks[out] = PipelineModels.POST_BATCH
            return out

        if ragged:
            if isinstance(tensor, tf.Tensor):
                assert (tensor.shape[0] is None)
                tensor = ragged_layers.pre_batch_ragged(tensor)
                tensor = batch(tensor, name=name)
                assert (tensor.shape[0] == self.batch_size)
                tensor = ragged_layers.post_batch_ragged(tensor)
                tensor.row_splits.set_shape((self.batch_size + 1,))
                return tensor
            else:
                assert (isinstance(tensor, tf.RaggedTensor))

        return batch(tensor, name=name)

    def batch(self,
              tensor: TensorLike,
              ragged: Optional[bool] = None,
              name: Optional[str] = None,
              partition: int = DataPartitions.FEATURES):
        DataPartitions.validate(partition)
        return self._batch(tensor, ragged=ragged, name=name)

    def trained_input(self,
                      tensor: TensorLike,
                      name: Optional[str] = None,
                      partition: int = DataPartitions.FEATURES
                     ) -> Optional[TensorLike]:
        if tensor.shape[0] != self.batch_size:
            raise ValueError(
                'batch_size not consistent with value provided in constructor. '
                'Expected {}, got shape {}'.format(self.batch_size,
                                                   tensor.shape))
        DataPartitions.validate(partition)
        self._marks[tensor] = PipelineModels.POST_BATCH
        assert (len(tensor.shape) > 0)
        self._post_batch_builder.add_output(tensor)
        self._output_partitions.append(partition)
        if partition == DataPartitions.FEATURES:
            out = self._trained_builder.add_input(to_spec(tensor))
            self._marks[out] = PipelineModels.TRAINED
            return out

    def trained_output(self, tensor: TensorLike) -> None:
        self._marks[tensor] = PipelineModels.TRAINED
        self._trained_builder.add_output(tensor)

    def build(self, **pipeline_kwargs) -> Tuple[BuiltPipeline, tf.keras.Model]:
        if self._use_cache:
            assert (len(self._pre_cache_builder._outputs) == len(
                self._pre_batch_builder._inputs))
        assert (len(self._pre_batch_builder._outputs) == len(
            self._post_batch_builder._inputs))

        pre_cache_model = _build_model(self._pre_cache_builder, 'pre_cache')
        pre_batch_model = _build_model(self._pre_batch_builder, 'pre_batch')
        post_batch_model = _build_model(self._post_batch_builder, 'post_batch')
        trained_model = self._trained_builder.build('trained')

        return BuiltPipeline(self.batch_size, pre_cache_model, pre_batch_model,
                             post_batch_model, tuple(self._base_partitions),
                             tuple(self._output_partitions,),
                             **pipeline_kwargs), trained_model


scope = Scope[PipelineBuilder](name='pipeline_builder')
get_default = scope.get_default


def base_input(tensor_spec: tf.TensorSpec,
               name: Optional[str] = None,
               partition: int = DataPartitions.FEATURES):
    return get_default().base_input(tensor_spec, name=name, partition=partition)


def cache(tensor: TensorLike, name: Optional[str] = None):
    return get_default().cache(tensor, name)


def trained_input(tensor: TensorLike, name: Optional[str] = None):
    return get_default().trained_input(tensor, name)


def trained_output(tensor: TensorLike):
    return get_default().trained_output(tensor)


def propagate_labels(tensor: TensorLike):
    return get_default().propagate_labels(tensor)


def propagate_weights(tensor: TensorLike):
    return get_default().propagate_weights(tensor)


def build() -> Tuple[BuiltPipeline, tf.keras.Model]:
    return get_default().build()


def batch(tensor: TensorLike,
          ragged: Optional[bool] = None,
          name: Optional[str] = None):
    return get_default().batch(tensor, ragged=ragged, name=name)


def propagate_marks(tensor: TensorLike) -> Optional[str]:
    return get_default().propagate_marks(tensor)


def get_batch_size() -> Optional[int]:
    return get_default().batch_size


get_mark = propagate_marks


def check_mark(tensor: TensorLike, mark: str, name: Optional[str] = None):
    return get_default().check_mark(tensor, mark, name)


def py_func_builder(pipeline_model: str = PipelineModels.PRE_BATCH,
                    name: Optional[str] = None):
    return get_default().py_func_builder(pipeline_model, name)


def _inputs(x: TensorLike) -> Tuple[tf.Tensor, ...]:
    if base_layer_utils.needs_keras_history(x):
        base_layer_utils.create_keras_history(x)
    inp = x._keras_history.layer.input
    if inp is x:
        return ()
    elif isinstance(inp, (tf.Tensor, tf.SparseTensor, tf.RaggedTensor)):
        return inp,
    else:
        return tuple(inp)
    # if isinstance(x, tf.Tensor):
    #     try:
    #         # return tuple(i for i in x.op.inputs
    #         #              if not base_layer_utils.needs_keras_history(i))
    #         # return tuple(
    #         #     i for i in x.op.inputs if i.name != 'keras_learning_phase:0')
    #         return tuple(x._keras_history.layer.input)
    #     except AttributeError:
    #         if tf.executing_eagerly():
    #             logging.info('Failed to get inputs in eager mode')
    #             return ()
    #         raise

    # elif isinstance(x, tf.RaggedTensor):
    #     return (x.flat_values,) + x.nested_row_splits
    # elif isinstance(x, tf.SparseTensor):
    #     return x.indices, x.values
    # else:
    #     raise ValueError('Invalid type of x: expected Tensor, RaggedTensor'
    #                      ' or SparseTensor, got {}'.format(x))


class Marks(object):

    def __init__(self):
        self._base: TensorDict[str] = TensorDict()

    def __getitem__(self, x: TensorLike) -> Optional[str]:
        return self._base.get(x, None)

    def __setitem__(self, x: TensorLike, mark: str) -> None:
        # check consistency
        m = self.propagate(x)
        if m is not None and m != mark:
            raise ValueError(
                'Attempted to mark x with {}, got inputs with mark {}'.format(
                    mark, m))
        # propagate marks down dependencies
        self._propagate_down(x, mark)

    def __contains__(self, x: TensorLike) -> bool:
        return x in self._base

    def _propagate_down(self, x: TensorLike, mark: str) -> None:
        if x not in self:
            self._base[x] = mark
            for i in _inputs(x):
                self._propagate_down(i, mark)

    def propagate(self, end: TensorLike) -> Optional[str]:
        mark = self._base.get(end)
        if mark is not None:
            return mark
        inputs = _inputs(end)
        if len(inputs) == 0:
            return None
        # get marks from all inputs, ensuring consistency
        for i in inputs:
            mi = self.propagate(i)
            if mi is not None:
                if mark is None:
                    mark = mi
                elif mi != mark:
                    raise ValueError(
                        'different marks detected: {} and {}'.format(mark, mi))
        if mark is None:
            return None

        # propagate mark back down input tree.
        self._propagate_down(end, mark)
        return mark


@gin.configurable(module='kb.framework')
def built_trainable(
        problem,
        builder_fn: Callable[[NestedTensorLikeSpec, NestedTensorLikeSpec, int],
                             PipelineBuilder],
        optimizer_fn: Optional[
            Callable[[], tf.keras.optimizers.Optimizer]] = None,
        model_dir: Optional[str] = None,
        **pipeline_kwargs):
    with problem_scope(problem):
        builder = builder_fn(problem.base_spec, problem.outputs_spec,
                             problem.get_pipeline('train').batch_size)
        pipeline, model = builder.build(**pipeline_kwargs)
        if optimizer_fn is not None:
            optimizer = optimizer_fn()
            model.compile(optimizer=optimizer,
                          loss=problem.loss,
                          metrics=problem.metrics)
    problem = problem.rebuild(pipeline=pipeline)
    return Trainable(problem=problem, model=model, model_dir=model_dir)


@gin.configurable(module='kb.framework')
def learning_cond(args, true_fn, false_fn, **kwargs):
    # assert (not tf.executing_eagerly())
    if isinstance(args, tf.Tensor):
        args = args,
    phase = tf.keras.backend.learning_phase()
    if isinstance(phase, tf.Tensor):
        return tf.cond(
            phase,
            lambda: true_fn(*args, **kwargs),
            lambda: false_fn(*args, **kwargs),
        )
    elif isinstance(phase, bool):
        return true_fn(*args, **kwargs) if phase else false_fn(*args, **kwargs)
    else:
        raise ValueError('No learning_phase defined')
