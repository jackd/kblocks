from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Callable

import tensorflow as tf
from kblocks.framework.pipelines.builder.py_func_builder import PyFuncBuilder
from kblocks.tf_typing import TensorLike
from kblocks.layers import Lambda


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

    def add_input(self, inp: TensorLike) -> None:
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
        if self._py_func_builder is None:
            return tf.keras.models.Model(self._inputs,
                                         self._outputs,
                                         name='simple_model')

        py_func_outputs = Lambda(self._py_func_builder.run)(
            self._py_func_builder.input_tensors)
        # Add dummy batch dimension to make keras models play nicely
        py_func_outputs = tf.nest.map_structure(
            lambda x: tf.expand_dims(x, axis=0), py_func_outputs)

        final_model = tf.keras.models.Model(
            self._inputs + list(self._py_func_builder.output_tensors),
            self._outputs,
            name='final_model')
        final_out = final_model(self._inputs + list(py_func_outputs))
        return tf.keras.models.Model(self._inputs,
                                     final_out,
                                     name='combined_model')
