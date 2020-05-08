import numpy as np
import tensorflow as tf
from tqdm import tqdm

example_size = 10 ** 7
dataset_size = 32
num_shards = 8
batch_size = 4


def gen():
    for _ in range(dataset_size):
        yield np.random.uniform(size=example_size)
        # yield np.zeros((example_size,))


dataset = tf.data.Dataset.from_generator(gen, tf.float64, (example_size,))

base = dataset.cache("/tmp/cache_test/base")


def profile(dataset, name):
    dataset = dataset.batch(batch_size).prefetch(-1)
    for _ in tqdm(
        dataset, total=dataset_size // batch_size, desc=f"{name}, initial run"
    ):
        pass

    for _ in tqdm(
        dataset, total=dataset_size // batch_size, desc=f"{name}, second run"
    ):
        pass


sharded = [
    dataset.shard(num_shards, i).cache(f"/tmp/cache_test/shard-{i}")
    for i in range(num_shards)
]
sharded = tf.data.Dataset.from_tensor_slices(sharded).interleave(lambda x: x)

# both run at ~4 hz
profile(sharded, "sharded")
profile(base, "base")
