from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kblocks.tf_typing import TensorLike, TensorLikeSpec, NestedTensorSpec, NestedTensorLike
from typing import Callable


def to_spec(tensor: TensorLike) -> TensorLikeSpec:
    """Convert a (Ragged/Sparse)Tensor to the corresponding TensorSpec."""
    if isinstance(tensor, tf.RaggedTensor):
        return tf.RaggedTensorSpec.from_value(tensor)
    elif isinstance(tensor, tf.Tensor):
        return tf.TensorSpec.from_tensor(tensor)
    elif isinstance(tensor, tf.SparseTensor):
        return tf.SparseTensorSpec.from_value(tensor)
    else:
        raise TypeError('Expected TensorLikeSpec, got {}'.format(tensor))


def map_spec(map_fn: Callable[[NestedTensorLike], NestedTensorLike],
             spec: NestedTensorSpec) -> NestedTensorSpec:
    """
    Get the specification corresponding to a spec transformation.

    Args:
        map_fn: function applied.
        spec: possibly nested input spec structure.

    Returns:
        possibly nested output spec structure corresponding to the spec of
        map_fn applied to tensors corresponding to input spec.
    """

    def gen():
        raise NotImplementedError()

    dataset = tf.data.Dataset.from_generator(
        gen, tf.nest.map_structure(lambda spec: spec.dtype),
        tf.nest.map_structure(lambda spec: spec.shape))
    dataset = dataset.map(map_fn)
    return dataset.spec
