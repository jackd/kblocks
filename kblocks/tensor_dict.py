import collections
from typing import Generic, TypeVar

import tensorflow as tf

from kblocks.tf_typing import TensorLike

RaggedComponents = collections.namedtuple(
    "RaggedComponents", ["flat_values", "nested_row_splits"]
)

SparseComponents = collections.namedtuple("SparseComponents", ["indices", "values"])

T = TypeVar("T")


def _tensor_key(x):
    if isinstance(x, tf.RaggedTensor):
        return RaggedComponents(
            x.flat_values.experimental_ref(),
            tuple(rs.experimental_ref() for rs in x.nested_row_splits),
        )
    if isinstance(x, tf.SparseTensor):
        return SparseComponents(
            x.indices.experimental_ref(), x.values.experimental_ref()
        )
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return x.experimental_ref()

    raise KeyError(
        "x must be a Tensor, Variable, SparseTensor or RaggedTensor, got {}".format(x)
    )


class TensorDict(Generic[T], collections.MutableMapping[TensorLike, T]):
    def __init__(self, **kwargs):
        self._base = {}
        self._compounds = {}
        self.update(**kwargs)

    def __getitem__(self, key: TensorLike) -> T:
        key_ref = _tensor_key(key)
        return self._base[key_ref]

    def __setitem__(self, key: TensorLike, value: T) -> None:
        key_ref = _tensor_key(key)
        self._base[key_ref] = value

        if not isinstance(key, tf.Tensor):
            self._compounds[key_ref] = key

    def __delitem__(self, key: TensorLike):
        key_ref = _tensor_key(key)
        del self._base[key_ref]
        if not isinstance(key, tf.Tensor):
            del self._compounds[key_ref]

    def __iter__(self):
        def gen():
            for k in self._base:
                if isinstance(k, (RaggedComponents, SparseComponents)):
                    yield self._compounds[k]
                else:
                    yield k._wrapped  # pylint:disable=protected-access

        return iter(gen())

    def __len__(self) -> int:
        return len(self._base)

    def __contains__(self, key: TensorLike) -> bool:
        return _tensor_key(key) in self._base
