import tensorflow_datasets as tfds
# from kblocks.framework.problems

import numpy as np
import tensorflow as tf

TensorInfo = tfds.core.features.TensorInfo


class SparseTensor(tfds.core.features.FeatureConnector):

    def __init__(self,
                 dtype,
                 ndims=None,
                 dense_shape=None,
                 size=None,
                 indices_dtype=tf.int64,
                 dense_shape_dtype=tf.int64):
        self._dtype = dtype
        self._indices_dtype = indices_dtype
        self._dense_shape_dtype = dense_shape_dtype
        if dense_shape is None:
            if ndims is None:
                raise ValueError('one of ndims or dense_shape is required.')
            else:
                ndims = len(dense_shape)
        elif ndims is not None:
            assert (len(dense_shape) == ndims)

        assert (isinstance(ndims, int))
        self._ndims = ndims
        if dense_shape is not None:
            dense_shape = tuple(dense_shape)
            if any(d is None for d in dense_shape):
                raise ValueError(
                    'all entries of dense_shape must be given if not None')
        self._dense_shape = dense_shape
        assert (isinstance(size, int) or size is None)
        self._size = size

    def get_tensor_info(self):
        out = {
            'indices':
                TensorInfo(shape=(self._size, self._ndims),
                           dtype=self._indices_dtype),
            'values':
                TensorInfo(shape=(self._size,), dtype=self._dtype)
        }
        if self._dense_shape is None:
            out['dense_shape'] = TensorInfo(shape=(self._ndims,),
                                            dtype=self._dense_shape_dtype)
        return out

    def encode_example(self, example_data):
        data = {
            'indices':
                np.array(example_data.indices, dtype=self._indices_dtype),
            'values':
                np.array(example_data.values, dtype=self._dtype),
        }
        if self._dense_shape is None:
            data['dense_shape'] = np.array(example_data.dense_shape,
                                           dtype=self._dense_shape_dtype)
        return data

    def decode_example(self, tfexample_data):
        dense_shape = (tfexample_data['dense_shape']
                       if self._dense_shape is None else self._dense_shape)
        return tf.SparseTensor(indices=tfexample_data['indices'],
                               values=tfexample_data['values'],
                               dense_shape=dense_shape)


class RaggedTensor(tfds.core.features.FeatureConnector):

    def __init__(self, dtype, row_splits_dtype, flat_values_shape, ragged_rank):
        self._dtype = dtype
        self._row_splits_dtype = row_splits_dtype
        self._flat_values_shape = flat_values_shape
        self._ragged_rank = ragged_rank

    def get_tensor_info(self):
        flat_values = TensorInfo(dtype=self._dtype,
                                 shape=self._flat_values_shape)
        nested_row_splits = tuple(
            TensorInfo(dtype=self._row_splits_dtype, shape=(None,))
            for _ in range(self._ragged_rank))
        return dict(flat_values=flat_values,
                    nested_row_splits=nested_row_splits)

    def encode_example(self, example_data):
        return dict(flat_values=np.array(example_data.flat_values,
                                         dtype=self._dtype),
                    nested_row_splits=tuple(
                        np.array(rs, dtype=self._row_splits_dtype)
                        for rs in example_data.nested_row_splits))

    def decode_example(self, tfexample_data):
        return tf.RaggedTensor.from_nested_row_splits(**tfexample_data)
