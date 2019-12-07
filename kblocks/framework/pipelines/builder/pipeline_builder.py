from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional

from absl import logging
import tensorflow as tf

from kblocks.ops import ragged as ragged_ops
from kblocks.spec import to_spec
from kblocks.scope import Scope
from kblocks.tensor_dict import TensorDict
from kblocks.framework.pipelines.builder.model_builder import ModelBuilder
from kblocks.framework.pipelines.builder.utils import assert_is_tensor_spec
from kblocks.framework.pipelines.core import Pipeline

from kblocks.tf_typing import NestedTensorLike
from kblocks.tf_typing import NestedTensorLikeSpec
from kblocks.tf_typing import TensorLike
from typing import Optional, Tuple
Lambda = tf.keras.layers.Lambda
Input = tf.keras.layers.Input

Model = tf.keras.Model


class BuiltPipeline(Pipeline):

    def __init__(self, pre_batch_model: Model, post_batch_model: Model,
                 trained_model: Model):
        if pre_batch_model.trainable_weights:
            raise ValueError(
                'pre_batch_model should not have any trainable weights')
        if post_batch_model.trainable_weights:
            raise ValueError(
                'post_batch_model should not have any trainable weights')
        self._pre_batch_model = pre_batch_model
        self._post_batch_model = post_batch_model
        self._trained_model = trained_model

    def pre_batch_map(self, *args: NestedTensorLike) -> NestedTensorLike:
        """Mapping applied to dataset features before batching."""
        args = [tf.expand_dims(arg, axis=0) for arg in tf.nest.flatten(args)]
        out = self._pre_batch_model(args)
        if isinstance(out, list):
            out = tuple(out)
        # reconstruct ragged structure here.
        return out

    def post_batch_map(self, *args: NestedTensorLike) -> NestedTensorLike:
        """Mapping applied to dataset features after batching."""
        out = self._post_batch_model(args)
        if isinstance(out, list):
            out = tuple(out)

        return out

    @property
    def features_spec(self) -> NestedTensorLikeSpec:
        """Spec for pre-prebatch features."""
        return tf.nest.map_structure(lambda x: to_spec(tf.squeeze(x, axis=0)),
                                     self._pre_batch_model.inputs)

    @property
    def model(self) -> Model:
        """`tf.keras.Model`."""
        return self._trained_model


class PipelineModels(object):
    PRE_BATCH = 'pre_batch'
    POST_BATCH = 'post_batch'
    TRAINED = 'trained'

    @classmethod
    def validate(cls, id_: str):
        if id_ not in cls.all():
            raise ValueError('invalid PipelineModel key {}'.format(id_))

    @classmethod
    def all(cls):
        return (PipelineModels.PRE_BATCH, PipelineModels.POST_BATCH,
                PipelineModels.TRAINED)


class PipelineBuilder(object):

    def __init__(self, batch_size: Optional[int] = None):
        self._batch_size = batch_size
        self._pre_batch_builder = ModelBuilder()
        self._post_batch_builder = ModelBuilder()
        self._trained_builder = ModelBuilder()
        self._builders = {
            PipelineModels.PRE_BATCH: self._pre_batch_builder,
            PipelineModels.POST_BATCH: self._post_batch_builder,
            PipelineModels.TRAINED: self._trained_builder,
        }
        self._marks = Marks()
        self._batch_size = batch_size

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    def propagate_marks(self, end: TensorLike) -> Optional[str]:
        return self._marks.propagate(end)

    def check_mark(self, end: TensorLike, mark: str, name: str = 'end') -> None:
        actual = self.propagate_marks(end)
        if actual != mark:
            raise RuntimeError(
                'Expected {} to have mark {}, but {} has mark {}'.format(
                    name, mark, end, actual))

    def py_func_builder(self,
                        pipeline_model: str = PipelineModels.PRE_BATCH,
                        name: Optional[str] = None):

        def callback(tensor):
            self._marks[tensor] = pipeline_model

        return self._builders[pipeline_model].py_func_builder(
            name, input_callback=callback, output_callback=callback)

    def pre_batch_input(self,
                        tensor_spec: tf.TensorSpec,
                        name: Optional[str] = None) -> tf.TensorSpec:
        assert_is_tensor_spec(tensor_spec)
        inp = Input(shape=tensor_spec.shape,
                    dtype=tensor_spec.dtype,
                    batch_size=1,
                    name=name)
        self._pre_batch_builder.add_input(inp)
        self._marks[inp] = PipelineModels.PRE_BATCH
        return tf.squeeze(inp, axis=0)

    def batch(self,
              tensor: TensorLike,
              ragged: Optional[bool] = None,
              name: Optional[str] = None) -> TensorLike:
        self._marks[tensor] = PipelineModels.PRE_BATCH
        if ragged is None:
            if isinstance(tensor, tf.RaggedTensor):
                ragged = True
            elif isinstance(tensor, tf.Tensor):
                if tensor.shape.ndims > 0 and tensor.shape[0] is None:
                    raise ValueError(
                        'ragged must be specified if leading dimension is None')
                ragged = False
            else:
                raise ValueError(
                    'tensor must be a Tensor or RaggedTensor, got {}'.format(
                        tensor))
        assert (ragged is not None)
        if ragged:
            if isinstance(tensor, tf.RaggedTensor):
                self._pre_batch_builder.add_output(tensor)
                inp = Input(shape=tensor.shape,
                            ragged=True,
                            dtype=tensor.dtype,
                            name=name,
                            batch_size=self.batch_size)
                self._marks[inp] = PipelineModels.POST_BATCH
                self._post_batch_builder.add_input(inp)

                # we rebuild to make keras play nicely.
                components = Lambda(lambda i: [
                    tf.identity(i.flat_values), *(
                        tf.identity(rs) for rs in i.nested_row_splits)
                ])(inp)
                rebuilt = Lambda(
                    lambda c: tf.RaggedTensor.from_nested_row_splits(
                        c[0], c[1:]))(components)
                self._marks[rebuilt] = PipelineModels.POST_BATCH
                return rebuilt

            elif isinstance(tensor, tf.Tensor):
                assert (tensor.shape[0] is None)
                output = Lambda(ragged_ops.pre_batch_ragged)(tensor)
                self._pre_batch_builder.add_output(output)
                inp = Input(output.shape,
                            dtype=output.dtype,
                            ragged=True,
                            name=name,
                            batch_size=self.batch_size)
                assert (isinstance(inp, tf.RaggedTensor))
                self._marks[inp] = PipelineModels.POST_BATCH
                self._post_batch_builder.add_input(inp)

                rebuilt = Lambda(ragged_ops.post_batch_ragged)(inp)

                # def f(rt):
                #     assert (isinstance(rt, tf.RaggedTensor))
                #     return [
                #         tf.identity(rt.flat_values),
                #         *(tf.identity(rs) for rs in rt.nested_row_splits[1:])
                #     ]

                # assert (isinstance(inp, tf.RaggedTensor))
                # components = Lambda(f)(inp)
                # rebuilt = Lambda(
                #     lambda c: tf.RaggedTensor.from_nested_row_splits(
                #         c[0], c[1:]))(components)
                self._marks[rebuilt] = PipelineModels.POST_BATCH
                return rebuilt
            else:
                raise ValueError(
                    'tensor must be a Tensor or RaggedTensor, got {}'.format(
                        tensor))
        else:
            if not isinstance(tensor, tf.Tensor):
                raise ValueError(
                    'tensor must be a tensor if Ragged is False, got {}'.format(
                        tensor))
            self._pre_batch_builder.add_output(tensor)
            out = Input(shape=tensor.shape,
                        dtype=tensor.dtype,
                        name=name,
                        batch_size=self.batch_size)
            self._post_batch_builder.add_input(out)
            self._marks[out] = PipelineModels.POST_BATCH
            return out

    def trained_input(self, tensor: TensorLike) -> TensorLike:
        if tensor.shape[0] != self.batch_size:
            raise ValueError(
                'batch_size not consistent with value provided in constructor. '
                'Expected {}, got shape {}'.format(self.batch_size,
                                                   tensor.shape))

        self._marks[tensor] = PipelineModels.POST_BATCH
        assert (len(tensor.shape) > 0)
        self._post_batch_builder.add_output(tensor)
        ragged = isinstance(tensor, tf.RaggedTensor)
        sparse = isinstance(tensor, tf.SparseTensor)
        inp = Input(shape=tensor.shape[1:],
                    dtype=tensor.dtype,
                    batch_size=tensor.shape[0],
                    ragged=ragged,
                    sparse=sparse)

        # TODO: consider using (rt|sp)._to_components() ?
        src_components: List[tf.Tensor]
        dst_components: List[tf.Tensor]
        if ragged:
            src_components = [tensor.flat_values, *tensor.nested_row_splits]
            dst_components = [inp.flat_values, *inp.nested_row_splits]
            assert (len(src_components) == len(dst_components))
            for src, dst in zip(src_components, dst_components):
                dst.set_shape(src.shape)
            components = Lambda(lambda x: tuple(
                tf.identity(c) for c in (x.flat_values, *x.nested_row_splits)))(
                    inp)
            out = Lambda(lambda x: tf.RaggedTensor.from_nested_row_splits(
                x[0], x[1:], validate=False))(components)

        elif sparse:
            src_components = [tensor.indices, tensor.values]
            dst_components = [inp.indices, inp.values]
            for src, dst in zip(src_components, dst_components):
                dst.set_shape(src.shape)
            components = Lambda(lambda x:
                                (tf.identity(x.indices), tf.identity(x.values),
                                 tf.identity(x.dense_shape)))(inp)
            out = Lambda(lambda x: tf.SparseTensor(*x))(components)
        else:
            assert (isinstance(tensor, tf.Tensor))
            out = inp

        self._marks[out] = PipelineModels.TRAINED
        self._trained_builder.add_input(inp)
        if (tuple(out.shape) != tuple(tensor.shape)):
            print(out.shape)
            print(tensor.shape)
            raise Exception()
        return out

    # def _trained_input(self, tensor: tf.Tensor) -> tf.Tensor:
    #     self._marks[tensor] = PipelineModels.POST_BATCH
    #     assert (isinstance(tensor, tf.Tensor))
    #     assert (len(tensor.shape) > 0)
    #     self._post_batch_builder.add_output(tensor)
    #     inp = Input(shape=tensor.shape[1:],
    #                 dtype=tensor.dtype,
    #                 batch_size=tensor.shape[0])
    #     self._marks[inp] = PipelineModels.TRAINED
    #     self._trained_builder.add_input(inp)
    #     return inp

    # def trained_input(self, tensor: TensorLike) -> TensorLike:
    #     if isinstance(tensor, tf.RaggedTensor):
    #         components = Lambda(
    #             lambda x: [tf.identity(x.flat_values)] +
    #             [tf.identity(rs) for rs in x.nested_row_splits])(tensor)
    #         components = [self._trained_input(c) for c in components]
    #         # return components
    #         rt = Lambda(lambda args: tf.RaggedTensor.from_nested_row_splits(
    #             args[0], args[1:]))(components)
    #         return rt
    #     elif not isinstance(tensor, tf.Tensor):
    #         raise ValueError('tensor must be a Tensor or RaggedTensor, got '
    #                          '{}'.format(tensor))
    #     if len(tensor.shape) == 0:
    #         tensor = tf.expand_dims(tensor, axis=0)
    #         tensor = self._trained_input(tensor)
    #         return tf.squeeze(tensor, axis=0)
    #     return self._trained_input(tensor)

    def trained_output(self, tensor: TensorLike) -> TensorLike:
        self._marks[tensor] = PipelineModels.TRAINED
        self._trained_builder.add_output(tensor)
        return tensor

    def build(self) -> BuiltPipeline:
        prebatch = self._pre_batch_builder.build()
        postbatch = self._post_batch_builder.build()
        trained = self._trained_builder.build()
        return BuiltPipeline(prebatch, postbatch, trained)


scope = Scope[PipelineBuilder](name='pipeline_builder')
get_default = scope.get_default


def pre_batch_input(tensor_spec: tf.TensorSpec):
    return get_default().pre_batch_input(tensor_spec)


def trained_input(tensor: TensorLike):
    return get_default().trained_input(tensor)


def trained_output(tensor: TensorLike):
    return get_default().trained_output(tensor)


def build():
    return get_default().build()


def batch(tensor: TensorLike, ragged: Optional[bool] = None):
    return get_default().batch(tensor, ragged=ragged)


def propagate_marks(tensor: TensorLike) -> Optional[str]:
    return get_default().propagate_marks(tensor)


def get_batch_size() -> Optional[int]:
    return get_default().batch_size


get_mark = propagate_marks


def check_mark(tensor: TensorLike, mark: str, name: Optional[str]=None):
    return get_default().check_mark(tensor, mark, name)


def py_func_builder(pipeline_model: str = PipelineModels.PRE_BATCH,
                    name: Optional[str] = None):
    return get_default().py_func_builder(pipeline_model, name)


def _inputs(x: TensorLike) -> Tuple[tf.Tensor, ...]:
    if isinstance(x, tf.Tensor):
        try:
            return tuple(x.op.inputs)
        except AttributeError:
            if tf.executing_eagerly():
                logging.info('Failed to get inputs in eager mode')
                return ()
            raise

    elif isinstance(x, tf.RaggedTensor):
        return (x.flat_values,) + x.nested_row_splits
    elif isinstance(x, tf.SparseTensor):
        return x.indices, x.values
    else:
        raise ValueError('Invalid type of x: expected Tensor, RaggedTensor'
                         ' or SparseTensor, got {}'.format(x))


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
        mark = None
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
