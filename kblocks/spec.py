from typing import Callable

import tensorflow as tf

from kblocks.tf_typing import (
    NestedTensorLike,
    NestedTensorSpec,
    TensorLike,
    TensorLikeSpec,
)


def to_input(spec: TensorLikeSpec) -> TensorLike:
    if isinstance(spec, tf.TensorSpec):
        return tf.keras.Input(
            shape=spec.shape[1:], batch_size=spec.shape[0], dtype=spec.dtype
        )
    if isinstance(spec, tf.RaggedTensorSpec):
        return tf.keras.Input(
            shape=spec._shape[1:],  # pylint:disable=protected-access
            batch_size=spec._shape[0],  # pylint:disable=protected-access
            dtype=spec._dtype,  # pylint:disable=protected-access
            ragged=True,
        )
    if isinstance(spec, tf.SparseTensorSpec):
        return tf.keras.Input(
            shape=spec.shape[1:],
            batch_size=spec.shape[0],
            dtype=spec.dtype,
            sparse=True,
        )
    raise TypeError("Unrecognized spec type {}".format(type(spec)))


def to_spec(tensor: TensorLike) -> TensorLikeSpec:
    """Convert a (Ragged/Sparse)Tensor to the corresponding TensorSpec."""
    if isinstance(tensor, tf.RaggedTensor):
        return tf.RaggedTensorSpec.from_value(tensor)
    if isinstance(tensor, tf.Tensor):
        return tf.TensorSpec.from_tensor(tensor)
    if isinstance(tensor, tf.SparseTensor):
        return tf.SparseTensorSpec.from_value(tensor)
    raise TypeError("Expected TensorLikeSpec, got {}".format(tensor))


def map_spec(
    map_fn: Callable[[NestedTensorLike], NestedTensorLike], spec: NestedTensorSpec
) -> NestedTensorSpec:
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
        gen,
        tf.nest.map_structure(dtype, spec),
        tf.nest.map_structure(shape, spec),
    )
    dataset = dataset.map(map_fn)
    return dataset.element_spec


def shape(spec: TensorLikeSpec) -> tf.TensorShape:
    if isinstance(spec, tf.RaggedTensor):
        return spec._shape  # pylint:disable=protected-access
    if isinstance(spec, (tf.TensorSpec, tf.SparseTensorSpec)):
        return spec.shape
    raise ValueError(f"Expected TensorLikeSpec, got {spec}")


def dtype(spec: TensorLikeSpec) -> tf.DType:
    if isinstance(spec, tf.RaggedTensor):
        return spec._dtype  # pylint:disable=protected-access
    if isinstance(spec, (tf.TensorSpec, tf.SparseTensorSpec)):
        return spec.dtype
    raise ValueError(f"Expected TensorLikeSpec, got {spec}")
