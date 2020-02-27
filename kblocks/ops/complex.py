import tensorflow as tf


def swish(x: tf.Tensor, threshold=-4.):
    return tf.where(x < threshold, tf.zeros_like(x), x * tf.nn.sigmoid(x))


def modified_swish(x: tf.Tensor, threshold=-4.):
    if x.dtype.is_complex:
        # x * sigmoid(real(x))
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        sig = tf.nn.sigmoid(real)
        return tf.where(real < threshold, tf.zeros_like(x),
                        tf.complex(real * sig, imag * sig))
    else:
        return swish(x)


def complex_relu(x: tf.Tensor):
    return tf.complex(tf.nn.relu(tf.math.real(x)), tf.nn.relu(tf.math.imag(x)))


def softplus(x: tf.Tensor, threshold=5.):
    return tf.where(x > threshold, x, tf.math.log(1 + tf.math.exp(x)))
