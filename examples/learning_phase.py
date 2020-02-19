import tensorflow as tf

dataset = tf.data.Dataset.range(10)


def map_fn(x):
    return tf.keras.backend.in_train_phase(x, 2 * x)


inp = tf.keras.Input(shape=(), dtype=tf.int64)
out = tf.keras.layers.Lambda(map_fn)(tf.squeeze(inp, axis=0))
model = tf.keras.Model(inp, out)


def train_map_fn(x):
    with tf.keras.backend.learning_phase_scope(True):
        return model(tf.expand_dims(x, axis=0))


def test_map_fn(x):
    with tf.keras.backend.learning_phase_scope(False):
        return model(tf.expand_dims(x, axis=0))


# for example in dataset.map(train_map_fn):
#     print(example.numpy())
# for example in dataset.map(test_map_fn):
#     print(example.numpy())

with tf.keras.backend.learning_phase_scope(False):
    test_dataset = dataset.map(lambda x: model(tf.expand_dims(x, axis=0)))
for example in test_dataset:
    print(example.numpy())
