import functools
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from image_utils import augment_image_example, get_augmented_data, simple_cnn

from kblocks.experiments.fit import Fit, logging_callbacks
from kblocks.tf_config import TfConfig

logging.set_verbosity(logging.INFO)

TfConfig(deterministic_ops=True, seed=0, global_rng_seed=0).configure()

name = "cifar10"
num_classes = 10
batch_size = 32
shuffle_buffer = 256
epochs = 5
run = 0
experiment_dir = os.path.join("/tmp/kblocks/simple", name, "base", f"run-{run:03d}")
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
optimizer = tf.keras.optimizers.Adam()

tfds_kwargs = dict(name=name, as_supervised=True)

train_data = get_augmented_data(
    tfds.load(
        split="train",
        shuffle_files=True,
        read_config=tfds.core.utils.read_config.ReadConfig(
            shuffle_seed=0, shuffle_reshuffle_each_iteration=True
        ),
        **tfds_kwargs,
    ),
    shuffle_buffer=shuffle_buffer,
    shuffle_seed=0,
    augment_seed=0,
    batch_size=batch_size,
    map_func=functools.partial(augment_image_example, noise_stddev=0.05),
    use_stateless_map=True,
)

validation_data = get_augmented_data(
    tfds.load(split="test", **tfds_kwargs),
    batch_size=batch_size,
    map_func=augment_image_example,
)

model = simple_cnn(train_data.dataset.element_spec[0], num_classes)
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

callbacks = logging_callbacks(experiment_dir)

fit = Fit(
    experiment_dir=experiment_dir,
    epochs=5,
    train_data=train_data,
    validation_data=validation_data,
    callbacks=callbacks,
    model=model,
)
fit.run()
