import tensorflow as tf

from kblocks.data.transforms import ShuffleRng

tf.random.get_global_generator().reset_from_seed(0)

dataset = tf.data.Dataset.range(10)
shuffle = ShuffleRng(5)
dataset = dataset.apply(shuffle)
print([el.numpy() for el in dataset])
print([el.numpy() for el in dataset])
