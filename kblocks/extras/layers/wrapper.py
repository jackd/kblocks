from typing import Sequence
import functools
import numpy as np
import tensorflow as tf

# from tensorflow.python.keras.engine import base_layer_utils  # pylint: disable=no-name-in-module,import-error


def partition(data: Sequence, partitions, num_partitions=None):
    if len(data) != len(partitions):
        raise ValueError(
            "lengths of data and partitions must match, but {} != {}".format(
                len(data), len(partitions)
            )
        )
    if num_partitions is None:
        num_partitions = np.max(partitions) + 1
    out = [[] for _ in range(num_partitions)]
    for d, p in zip(data, partitions):
        out[p].append(d)
    return out


def stitch(indices, data: Sequence[Sequence]):
    """Inverse of partition."""
    out = []
    counts = np.zeros((len(data),), dtype=np.int64)
    for i in indices:
        out.append(data[i][counts[i]])
        counts[i] += 1
    for c, d in zip(counts, data):
        if c != len(d):
            raise ValueError("Not all data elements consumed")
    return out


def is_keras_tensor(x):
    return isinstance(
        x, (tf.Tensor, tf.Variable, tf.SparseTensor, tf.RaggedTensor)
    )  # and hasattr(x, '_keras_history')


def _as_lambda(keras_args, fn, non_keras_args, structure, partition):
    flat_args = stitch(partition, (non_keras_args, keras_args))
    args, kwargs = tf.nest.pack_sequence_as(structure, flat_args)
    return fn(*args, **kwargs)


def as_lambda(fn, *args, **kwargs):
    structure = tf.nest.map_structure(lambda x: True, (args, kwargs))
    flat_args = tf.nest.flatten((args, kwargs))

    is_tensor = np.array([is_keras_tensor(x) for x in flat_args], dtype=np.uint8)

    non_keras_args, keras_args = partition(flat_args, is_tensor, 2)
    # if layer.name in (
    #         # 'lambda_1',
    #         'lambda_6',):
    print("---")
    print(structure)
    print(non_keras_args)
    print(is_tensor)
    print(keras_args)
    # raise Exception()
    layer = tf.keras.layers.Lambda(
        _as_lambda,
        arguments=dict(
            structure=structure,
            non_keras_args=non_keras_args,
            partition=is_tensor,
            fn=fn,
        ),
    )
    return layer(keras_args)


def wrap_as_lambda(fn):
    out = functools.update_wrapper(functools.partial(as_lambda, fn=fn), fn)
    out.__doc__ = "Lambda wrapped version\n{}".format(out.__doc__)
    return out
