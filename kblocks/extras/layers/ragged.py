"""
tf.keras.layers.Lambda wrappers around RaggedComponent constructors/components.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Iterable, Union, Tuple, NamedTuple, Optional
import tensorflow as tf
from kblocks.ops import ragged as ragged_ops
from kblocks.tf_typing import Dimension

RTensor = Union[tf.Tensor, tf.RaggedTensor]


class RaggedComponents(NamedTuple):
    flat_values: tf.Tensor
    nested_row_splits: Tuple[tf.Tensor, ...]


Lambda = tf.keras.layers.Lambda

# def Lambda(*args, **kwargs):
#     layer = tf.keras.layers.Lambda(*args, **kwargs)
#     # print(layer.name)
#     if layer.name == 'lambda_27':
#         raise Exception()
#     return layer


def from_row_splits(values: RTensor,
                    row_splits: tf.Tensor,
                    name=None,
                    validate=True) -> tf.RaggedTensor:
    return Lambda(lambda args: tf.RaggedTensor.from_row_splits(
        *args, name=name, validate=validate))([values, row_splits])


def from_nested_row_splits(flat_values: tf.Tensor,
                           nested_row_splits: Iterable[tf.Tensor],
                           name=None,
                           validate=True) -> tf.RaggedTensor:
    return Lambda(lambda args: tf.RaggedTensor.from_nested_row_splits(
        args[0], args[1:], name=name, validate=validate))(
            [flat_values, *nested_row_splits])


def from_row_lengths(values: tf.Tensor,
                     row_lengths: tf.Tensor,
                     name=None,
                     validate=True) -> tf.RaggedTensor:
    return Lambda(lambda args: tf.RaggedTensor.from_row_lengths(
        *args, name=name, validate=validate))([values, row_lengths])


def from_value_rowids(values: tf.Tensor,
                      rowids: tf.Tensor,
                      nrows=None,
                      name=None,
                      validate=True):

    args = [values, rowids]
    kwargs = dict(name=name, validate=validate)
    if nrows is not None:
        if isinstance(nrows, int):
            kwargs['nrows'] = nrows
        else:
            args.append(nrows)
    return Lambda(
        lambda args: tf.RaggedTensor.from_value_rowids(*args, **kwargs))(args)


def from_tensor(tensor: tf.Tensor,
                lengths: Optional[tf.Tensor] = None,
                padding: Optional[Union[tf.Tensor, int, float]] = None,
                ragged_rank: int = 1,
                name=None,
                row_splits_dtype=tf.int64):
    kwargs = dict(name=name,
                  ragged_rank=ragged_rank,
                  row_splits_dtype=row_splits_dtype)
    args = []
    names = []
    for name, value in (
        ('tensor', tensor),
        ('lengths', lengths),
        ('padding', padding),
    ):
        if isinstance(value, tf.Tensor):
            args.append(value)
            names.append(name)
        else:
            kwargs[name] = value

    def f(args, **kwargs):
        for name, value in zip(names, args):
            kwargs[name] = value
        return tf.RaggedTensor.from_tensor(**kwargs)

    return Lambda(f, arguments=kwargs)(args)


def values(rt: tf.RaggedTensor) -> RTensor:
    return Lambda(lambda rt: tf.identity(rt.values))(rt)


def value_rowids(rt: tf.RaggedTensor) -> tf.Tensor:
    return Lambda(lambda rt: rt.value_rowids())(rt)


def flat_values(rt: tf.RaggedTensor) -> tf.Tensor:
    return Lambda(lambda rt: tf.identity(rt.flat_values))(rt)


def row_splits(rt: tf.RaggedTensor) -> tf.Tensor:
    return Lambda(lambda rt: tf.identity(rt.row_splits))(rt)


def row_lengths(rt: tf.RaggedTensor) -> tf.Tensor:
    return Lambda(lambda rt: rt.row_lengths())(rt)


def row_starts(rt: tf.RaggedTensor) -> tf.Tensor:
    return Lambda(lambda rt: rt.row_starts())(rt)


def nested_row_splits(rt: tf.RaggedTensor) -> Tuple[tf.Tensor, ...]:
    out = Lambda(lambda rt: [tf.identity(rs) for rs in rt.nested_row_splits])(
        rt)
    return (out,) if isinstance(out, tf.Tensor) else tuple(out)


def components(rt: tf.RaggedTensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
    out = Lambda(lambda rt: [
        tf.identity(rt.flat_values), *(tf.identity(rs) for rs in rt.row_splits)
    ])(rt)
    return RaggedComponents(out[0], tuple(out[1:]))


def with_values(rt: tf.RaggedTensor, values: RTensor):
    return Lambda(lambda args: args[0].with_values(args[1]))([rt, values])


def with_flat_values(rt: tf.RaggedTensor, flat_values: tf.Tensor):
    return Lambda(lambda args: args[0].with_flat_values(args[1]))(
        [rt, flat_values])


def with_row_splits_dtype(rt: tf.RaggedTensor, dtype: tf.DType):
    return Lambda(lambda rt: rt.with_row_splits_dtype(dtype))(rt)


def pre_batch_ragged(tensor: tf.Tensor) -> tf.RaggedTensor:
    return Lambda(ragged_ops.pre_batch_ragged)(tensor)


def post_batch_ragged(rt: tf.RaggedTensor) -> tf.RaggedTensor:
    return Lambda(ragged_ops.post_batch_ragged)(rt)


def lengths_to_splits(row_lengths: tf.Tensor) -> tf.Tensor:
    return Lambda(ragged_ops.lengths_to_splits)(row_lengths)


def ids_to_lengths(rowids: tf.Tensor, nrows: Optional[Dimension] = None):
    if isinstance(nrows, tf.Tensor):
        args = [rowids, nrows]
        kwargs = {}
    else:
        args = [rowids]
        kwargs = dict(nrows=nrows)
    return Lambda(lambda args: ragged_ops.ids_to_lengths(*args, **kwargs))(args)


def splits_to_lengths(row_splits: tf.Tensor) -> tf.Tensor:
    return Lambda(ragged_ops.splits_to_lengths)(row_splits)


def lengths_to_ids(row_lengths: tf.Tensor, dtype=tf.int64) -> tf.Tensor:
    return Lambda(ragged_ops.lengths_to_ids,
                  arguments=dict(dtype=dtype))(row_lengths)


def lengths_to_mask(row_lengths: tf.Tensor, size: Optional[Dimension] = None):
    if isinstance(size, tf.Tensor):
        return Lambda(lambda args: ragged_ops.lengths_to_mask(*args))(
            [row_lengths, size])
    else:
        return Lambda(ragged_ops.lengths_to_mask,
                      arguments=dict(size=size))(row_lengths)


def mask_to_lengths(mask: tf.Tensor) -> tf.Tensor:
    return Lambda(ragged_ops.mask_to_lengths)(mask)


def row_max(values: tf.Tensor, row_lengths: tf.Tensor, num_segments: Dimension,
            max_length: Dimension) -> tf.Tensor:
    args = [values, row_lengths]
    names = ['values', 'row_lengths']
    kwargs = dict()
    if isinstance(num_segments, tf.Tensor):
        args.append(num_segments)
        names.append('num_segments')
    else:
        kwargs['num_segments'] = num_segments
    if isinstance(max_length, tf.Tensor):
        args.append(max_length)
        names.append('max_length')
    else:
        kwargs['max_length'] = max_length
    return Lambda(lambda args: ragged_ops.row_max(
        **{k: v for k, v in zip(names, args)}, **kwargs))(args)


def row_sum(values: tf.Tensor, row_lengths: tf.Tensor, num_segments: Dimension,
            max_length: Dimension) -> tf.Tensor:
    args = [values, row_lengths]
    names = ['values', 'row_lengths']
    kwargs = dict()
    if isinstance(num_segments, tf.Tensor):
        args.append(num_segments)
        names.append('num_segments')
    else:
        kwargs['num_segments'] = num_segments
    if isinstance(max_length, tf.Tensor):
        args.append(max_length)
        names.append('max_length')
    else:
        kwargs['max_length'] = max_length
    return Lambda(lambda args: ragged_ops.row_sum(
        **{k: v for k, v in zip(names, args)}, **kwargs))(args)


def segment_sum(values: tf.Tensor, segment_ids: tf.Tensor,
                num_segments: Dimension):
    args = [values, segment_ids]
    if isinstance(num_segments, tf.Tensor):
        args.append(num_segments)
        kwargs = {}
    else:
        kwargs = dict(num_segments=num_segments)
    return Lambda(lambda args: ragged_ops.segment_sum(*args, **kwargs))(args)


def repeat_ranges(row_lengths: tf.Tensor,
                  maxlen: Optional[Dimension] = None) -> tf.Tensor:
    if isinstance(maxlen, int):
        args = [row_lengths]
        kwargs = dict(maxlen=maxlen)
    else:
        args = [row_lengths, maxlen]
        kwargs = {}
    return Lambda(lambda args: ragged_ops.repeat_ranges(*args, **kwargs))(args)


def to_tensor(rt: tf.RaggedTensor,
              ncols: Optional[Dimension] = None) -> tf.Tensor:
    if isinstance(ncols, int):
        args = [rt]
        kwargs = dict(ncols=ncols)
    else:
        args = [rt, ncols]
        kwargs = {}
    return Lambda(lambda args: ragged_ops.to_tensor(*args, **kwargs))(args)
