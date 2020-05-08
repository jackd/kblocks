import tensorflow as tf

Lambda = tf.keras.layers.Lambda


def indices(st: tf.SparseTensor) -> tf.Tensor:
    assert isinstance(st, tf.SparseTensor)
    return Lambda(lambda x: tf.identity(x.indices))(st)


def values(st: tf.SparseTensor) -> tf.Tensor:
    assert isinstance(st, tf.SparseTensor)
    return Lambda(lambda x: tf.identity(x.values))(st)


def dense_shape(st: tf.SparseTensor) -> tf.Tensor:
    assert isinstance(st, tf.SparseTensor)
    return Lambda(lambda x: tf.identity(x.dense_shape))(st)


def _sparse_tensor(args):
    return tf.SparseTensor(*args)


def SparseTensor(indices, values, dense_shape) -> tf.SparseTensor:
    return Lambda(_sparse_tensor)([indices, values, dense_shape])
