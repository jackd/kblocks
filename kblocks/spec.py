from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from typing import Union
from kblocks.tf_typing import TensorLike, TensorLikeSpec, NestedTensorSpec
from typing import Callable


# def to_spec(tensor: Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
#            ) -> Union[tf.TensorSpec, tf.RaggedTensorSpec, tf.SparseTensorSpec]:
def to_spec(tensor: TensorLike) -> TensorLikeSpec:
    if isinstance(tensor, tf.RaggedTensor):
        return tf.RaggedTensorSpec.from_value(tensor)
    elif isinstance(tensor, tf.Tensor):
        return tf.TensorSpec.from_tensor(tensor)
    elif isinstance(tensor, tf.SparseTensor):
        return tf.SparseTensorSpec.from_value(tensor)
    else:
        raise TypeError('Expected TensorLikeSpec, got {}'.format(tensor))


def map_spec(map_fn: Callable, spec: NestedTensorSpec) -> NestedTensorSpec:

    def gen():
        raise NotImplementedError()

    dataset = tf.data.Dataset.from_generator(
        gen, tf.nest.map_structure(lambda spec: spec.dtype),
        tf.nest.map_structure(lambda spec: spec.shape))
    dataset = dataset.map(map_fn)
    return dataset.spec


def specs_are_consistent(a: tf.TensorSpec, b: tf.TensorSpec):
    if a.dtype != b.dtype:
        return False
    if len(a.shape) != len(b.shape):
        return False
    for sa, sb in zip(a.shape, b.shape):
        if sa is not None and sb is not None and sa != sb:
            return False
    return True
