from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Sequence, Union

import tensorflow as tf
from kblocks.tf_typing import TensorLike, NestedTensorLike
from kblocks.tf_typing import TensorLikeSpec


def input_like(inp: TensorLike):
    return tf.keras.Input(ragged=isinstance(inp, tf.RaggedTensor),
                          shape=inp.shape[1:],
                          batch_size=inp.shape[0],
                          dtype=inp.dtype)


def assert_is_input(inp: tf.Tensor):
    if not hasattr(inp, '_keras_history'):
        raise ValueError(
            'inp must have some _keras_history, got {}'.format(inp))
    layer, node_index, tensor_index = inp._keras_history
    del node_index, tensor_index
    if not isinstance(layer, tf.keras.layers.InputLayer):
        raise ValueError(
            'inp must come from an InputLayer, got {} comes from {}'.format(
                inp, layer))
    if len(layer._inbound_nodes) > 1:
        raise ValueError('len(layer._inbound_nodes) == {} > 1 for input {} with'
                         ' layer {}'.format(len(layer._inbound_nodes), inp,
                                            layer))
    if layer._inbound_nodes and layer._inbound_nodes[0].inbound_layers:
        raise ValueError('layer._inbound_nodes[0].inbound_layers == {}'.format(
            layer._inbound_nodes[0].inbound_layers))


def assert_is_keras_tensor(tensor: TensorLike):
    if not tf.keras.backend.is_keras_tensor(tensor):
        raise ValueError('tensor {} is not a keras tensor'.format(tensor))


class ModelBuilder(object):

    def __init__(self):
        self._inputs: List[TensorLike] = []
        self._outputs: List[TensorLike] = []

    def add_input(self, spec: TensorLikeSpec) -> TensorLike:
        if isinstance(spec, tf.TensorSpec):
            assert (len(spec.shape) >= 1)
            inp = tf.keras.Input(shape=spec.shape[1:],
                                 batch_size=spec.shape[0],
                                 dtype=spec.dtype)
        elif isinstance(spec, tf.RaggedTensorSpec):
            batch_size = spec._shape[0]
            inp = tf.keras.Input(
                shape=spec._shape[1:],
                batch_size=batch_size,
                dtype=spec._dtype,
                ragged=True,
            )
            if inp.shape[0] is None and batch_size is not None:
                inp.row_splits.set_shape((batch_size + 1,))
            assert (inp.shape[0] == batch_size)
        elif isinstance(spec, tf.SparseTensorSpec):
            inp = tf.keras.Input(shape=spec.shape[1:],
                                 batch_size=spec.shape[0],
                                 dtype=spec.dtype,
                                 sparse=True)
        else:
            raise TypeError('Invalid spec type {}'.format(type(spec)))

        self._inputs.append(inp)
        return inp

    def add_output(self, tensor: TensorLike) -> None:
        # assert_is_keras_tensor(tensor)
        self._outputs.append(tensor)

    def build(self, name='built_model') -> Optional[tf.keras.models.Model]:
        if len(self._inputs) == 0:
            return None
        # these are checked in add_* methods, but tf.keras doesn't respect
        # a core premise of tf where tensors are immutable.
        for x in self._inputs:
            assert_is_input(x)
        # for x in self._outputs:
        #     assert_is_keras_tensor(x)

        return tf.keras.models.Model(self._inputs, self._outputs, name=name)

    @classmethod
    def apply(cls, model, args: NestedTensorLike
             ) -> Union[TensorLike, Sequence[TensorLike]]:
        args = tf.nest.flatten(args)
        return model(args)


class UnbatchedModelBuilder(ModelBuilder):

    def add_input(self, spec: tf.TensorSpec, name: Optional[str] = None):
        assert (isinstance(spec, tf.TensorSpec))
        kwargs = dict(batch_size=1, name=name)
        if isinstance(spec, tf.TensorSpec):
            inp = tf.keras.Input(shape=spec.shape, dtype=spec.dtype, **kwargs)
        elif isinstance(spec, tf.RaggedTensorSpec):
            inp = tf.keras.Input(shape=spec._shape,
                                 dtype=spec._dtype,
                                 ragged=True,
                                 **kwargs)
            inp.row_splits.set_shape((2,))
            assert (inp.shape[0] == 1)
        elif isinstance(spec, tf.SparseTensorSpec):
            inp = tf.keras.Input(shape=spec.shape,
                                 dtype=spec.dtype,
                                 sparse=True,
                                 **kwargs)
        else:
            raise TypeError('Invalid spec type {}'.format(type(spec)))

        self._inputs.append(inp)
        return tf.keras.layers.Lambda(tf.squeeze, arguments=dict(axis=0))(inp)

    @classmethod
    def apply(cls, model, args: NestedTensorLike
             ) -> Union[TensorLike, Sequence[TensorLike]]:
        args = tf.nest.flatten(args)
        args = [tf.expand_dims(a, axis=0) for a in args]
        return model(args)
