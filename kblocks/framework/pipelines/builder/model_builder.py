from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Callable

import tensorflow as tf
from kblocks.framework.pipelines.builder.py_func_builder import PyFuncBuilder
from kblocks.tf_typing import TensorLike
layers = tf.keras.layers


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


class ModelBuilder(object):

    def __init__(self):
        self._inputs: List[TensorLike] = []
        self._outputs: List[TensorLike] = []
        self._py_func_builder: Optional[PyFuncBuilder] = None

    def py_func_builder(
            self,
            name: Optional[str] = None,
            input_callback: Optional[Callable[[tf.Tensor], None]] = None,
            output_callback: Optional[Callable[[tf.Tensor], None]] = None
    ) -> PyFuncBuilder:
        if self._py_func_builder is not None:
            if self._py_func_builder.name == name:
                return self._py_func_builder
            raise NotImplementedError('Only single py_func_builder supported')
        self._py_func_builder = PyFuncBuilder(name=name,
                                              input_callback=input_callback,
                                              output_callback=output_callback)
        return self._py_func_builder

    def add_input(self, inp: tf.Tensor) -> None:
        assert (isinstance(inp, (tf.Tensor, tf.RaggedTensor)))
        assert_is_input(inp)
        # if hasattr(inp, 'name') and inp.name == 'input_63:0':
        #     raise Exception()
        self._inputs.append(inp)

    def add_output(self, tensor: TensorLike) -> None:
        if not isinstance(tensor, (tf.Tensor, tf.RaggedTensor)):
            raise ValueError(
                'tensor must be a Tensor or RaggedTensor, got {}'.format(
                    tensor))
        self._outputs.append(tensor)

    def build(self) -> Optional[tf.keras.Model]:
        if len(self._inputs) == 0:
            return None
        for x in self._inputs:
            # this could be no longer an input because keras doesn't respect
            # a core premise of tf where tensors are immutable.
            assert_is_input(x)
        if self._py_func_builder is None or len(
                self._py_func_builder.output_tensors) == 0:
            return tf.keras.models.Model(self._inputs,
                                         self._outputs,
                                         name='simple_model')

        for x in self._py_func_builder.output_tensors:
            assert_is_input(x)

        final_inputs = [input_like(x) for x in self._inputs]

        # if there's an error below, it's most likely caused by a cyclic
        # dependency, i.e. an input has a dependency on our of the outputs.
        pf_inputs_model = tf.keras.models.Model(
            self._inputs, self._py_func_builder.input_tensors)

        pf_inputs = pf_inputs_model(final_inputs)

        py_func_outputs = layers.Lambda(self._py_func_builder.run)(pf_inputs)
        # Add dummy batch dimension to make keras models play nicely
        py_func_outputs = tf.nest.map_structure(
            lambda x: tf.expand_dims(x, axis=0), py_func_outputs)

        tail_model = tf.keras.models.Model(
            self._inputs + list(self._py_func_builder.output_tensors),
            self._outputs,
            name='tail_model')
        final_out = tail_model(final_inputs + list(py_func_outputs))

        return tf.keras.models.Model(final_inputs,
                                     final_out,
                                     name='combined_model')

        # final_out = tail_model(self._inputs + list(py_func_outputs))
        # for x in self._inputs:
        #     # this could be no longer an input because keras doesn't respect
        #     # a core premise of tf where tensors are immutable.
        #     assert_is_input(x)
        # return tf.keras.models.Model(self._inputs,
        #                              final_out,
        #                              name='combined_model')
