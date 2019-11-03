from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Iterable, Union, Optional, Tuple


def assert_compatible(
        name: str,
        tensor: tf.Tensor,
        shape: Optional[Union[Tuple[int], tf.TensorShape]] = None,
        dtypes: Optional[Union[tf.DType, Iterable[tf.DType]]] = None):
    if shape is not None:
        tf_shape = tf.TensorShape(shape)
        if not tf_shape.is_compatible_with(tensor.shape):
            raise ValueError(
                '{}.shape {} is not compatible with expected shape {}'.format(
                    name, tensor.shape, tf_shape))
    if dtypes is not None:
        if isinstance(dtypes, tf.DType):
            if tensor.dtype != dtypes:
                raise ValueError('{}.dtype {} must be {}'.format(
                    name, tensor.dtype, dtypes))
        elif hasattr(dtypes, '__contains__'):
            if tensor.dtype not in dtypes:
                raise ValueError('{}.dtype {} must be in {}'.format(
                    name, tensor.dtype, str(dtypes)))
        else:
            raise ValueError('Unrecognized dtypes type {}'.format(dtypes))
