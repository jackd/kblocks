from typing import Any, Mapping, NamedTuple, Sequence, Tuple, TypeVar, Union

import tensorflow as tf

Tensor = tf.Tensor
TensorSpec = tf.TensorSpec
RaggedTensorSpec = tf.RaggedTensorSpec
SparseTensorSpec = tf.SparseTensorSpec

TensorLikeSpec = Union[TensorSpec, RaggedTensorSpec, SparseTensorSpec]
RaggedComponents = NamedTuple(
    "RaggedComponents",
    [("flat_values", Tensor), ("nested_row_splits", Tuple[Tensor, ...])],
)

SparseComponents = NamedTuple(
    "SparseComponents", [("indices", Tensor), ("values", Tensor)]
)

TensorComponents = Union[Tensor, RaggedComponents, SparseComponents]

TensorLike = Union[Tensor, tf.RaggedTensor, tf.SparseTensor]

NestedTensors = Union[Sequence["NestedTensors"], Mapping[Any, "NestedTensors"], Tensor]

NestedTensorLike = Union[
    Sequence["NestedTensors"], Mapping[Any, "NestedTensors"], TensorLike
]

NestedTensorComponents = Union[
    Sequence["NestedComponents"], Mapping[Any, "NestedComponents"], TensorComponents
]

NestedTensorLikeSpec = Union[
    Sequence["NestedTensorLikeSpec"],
    Mapping[Any, "NestedTensorLikeSpec"],
    TensorLikeSpec,
]

NestedTensorSpec = Union[
    Sequence["NestedTensorSpec"], Mapping[Any, "NestedTensorSpec"], TensorLikeSpec
]

TensorOrVariable = Union[tf.Tensor, tf.Variable]

Dimension = Union[int, tf.Tensor]  # tensors should be scalar ints
DenseShape = Tuple[Dimension, Dimension]

T = TypeVar("T")
