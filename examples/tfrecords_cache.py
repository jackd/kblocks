import numpy as np
import tensorflow as tf
from kblocks.framework.cache import TFRecordsCacheManager
from kblocks.framework.cache.core import dataset_iterator


def main():

    cache_dir = "/tmp/cache_test/cache_files"

    def gen():
        yield np.zeros((5,)), np.ones((4,))
        yield np.zeros((3,)), np.ones((7,))

    dataset = tf.data.Dataset.from_generator(
        gen, (tf.float32, tf.float32), ((None,), (None,))
    )

    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (tf.range(10), 10 - tf.range(10)))

    # dataset = tf.data.Dataset.from_tensor_slices(
    #     dict(x=tf.range(10), y=10 - tf.range(10)))

    manager = TFRecordsCacheManager(cache_dir=cache_dir)
    manager.clear()
    dataset = manager(dataset)
    for example in dataset_iterator(dataset, as_numpy=True):
        print(example)


if __name__ == "__main__":
    main()
    with tf.Graph().as_default():
        main()
