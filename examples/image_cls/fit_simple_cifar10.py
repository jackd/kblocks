import functools
import os

import tensorflow as tf
from absl import logging

from image_utils import augment_image_example, simple_cnn
from kblocks.data.sources import TfdsSource
from kblocks.tf_config import TfConfig
from kblocks.trainables import Fit, Trainable

logging.set_verbosity(logging.INFO)

TfConfig(deterministic_ops=True, seed=0, global_rng_seed=0).configure()
name = "cifar10"
num_classes = 10
batch_size = 32
shuffle_buffer = 256
epochs = 5
model_dir = os.path.join("/tmp/kblocks/simple", name, "base")
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
optimizer = tf.keras.optimizers.Adam()

tfds_kwargs = dict(name=name, shuffle_files=False, as_supervised=True)

train_source = (
    TfdsSource(split="train", **tfds_kwargs)
    .shuffle_rng(shuffle_buffer)
    .batch(batch_size)
    .map_rng(functools.partial(augment_image_example, noise_stddev=0.05))
    .prefetch()
)

validation_source = (
    TfdsSource(split="test", **tfds_kwargs)
    .batch(batch_size)
    .map(augment_image_example)
    .prefetch()
)

model = simple_cnn(train_source.dataset.element_spec[0], num_classes)
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
trainable = Trainable(
    model=model, train_source=train_source, validation_source=validation_source
)
fit = Fit(trainable, epochs=5, save_dir="/tmp/kblocks-examples/image-cls")
fit.run()
