import tensorflow as tf
from kblocks.extras.layers import as_lambda
from kblocks.extras.layers import wrap_as_lambda

try:
    values = tf.keras.Input(shape=(), dtype=tf.float32)
    row_lengths = tf.keras.Input(shape=(), dtype=tf.int32)
    rt = tf.RaggedTensor.from_row_lengths(values, row_lengths, validate=False)
    model = tf.keras.Model([values, row_lengths], rt)
    print("Succeeds without wrapper")
except Exception:
    print("Fails without wrapper")

try:
    values = tf.keras.Input(shape=(), dtype=tf.float32)
    row_lengths = tf.keras.Input(shape=(), dtype=tf.int32)
    rt = as_lambda(
        tf.RaggedTensor.from_row_lengths, values, row_lengths, None, False  # name
    )
    model = tf.keras.Model([values, row_lengths], rt)
    print("Succceeds with wrapper using only args")
except:
    print("Fails with wrapper using only args")

try:
    row_lengths = tf.keras.Input(shape=(), dtype=tf.int32)
    rt = as_lambda(
        tf.RaggedTensor.from_row_lengths,
        tf.range(100),  # not a keras input
        row_lengths=row_lengths,
        validate=False,
    )
    model = tf.keras.Model(row_lengths, rt)
    print("Succceeds with wrapper using args and kwargs")
except:
    print("Fails with wrapper using args and kwargs")


@wrap_as_lambda
def f(x, y):
    """Docs of the op function."""
    return x + y


print("f docs:")
print(f.__doc__)
