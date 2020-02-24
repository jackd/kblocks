import tensorflow as tf
from kblocks.framework.cache import TFRecordsCacheManager
from kblocks.framework.cache.core import dataset_iterator


def main():

    cache_dir = '/tmp/cache_test'

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.range(10), 10 - tf.range(10)))

    # dataset = tf.data.Dataset.from_tensor_slices(
    #     dict(x=tf.range(10), y=10 - tf.range(10)))

    manager = TFRecordsCacheManager(cache_dir=cache_dir, num_shards=5)
    manager.clear()
    dataset = manager(dataset)
    for example in dataset_iterator(dataset, as_numpy=True):
        print(example)


if __name__ == '__main__':
    main()
    with tf.Graph().as_default():
        main()
