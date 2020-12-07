import functools
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags, logging
from image_utils import augment_image_example, get_augmented_data, simple_cnn

from kblocks.experiments.fit import Fit, logging_callbacks
from kblocks.tf_config import TfConfig

flags.DEFINE_integer("batch_size", default=32, help="images per batch")
flags.DEFINE_integer("epochs", default=5, help="number of epochs to train for")
flags.DEFINE_integer("run", default=0, help="index of run")
flags.DEFINE_integer("shuffle_buffer", default=256, help="shuffle buffer size")
flags.DEFINE_integer("seed", default=0, help="seed used in various places")


def main(_):
    FLAGS = flags.FLAGS

    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    run = FLAGS.run
    shuffle_buffer = FLAGS.shuffle_buffer
    seed = FLAGS.seed

    TfConfig(deterministic_ops=True, seed=seed, global_rng_seed=seed).configure()

    name = "cifar10"
    num_classes = 10
    experiment_dir = os.path.join(
        "/tmp/kblocks/examples/image_cls", name, "base", f"run-{run:03d}"
    )
    logging.info(f"Experiment_dir = {experiment_dir}")
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
        epochs=epochs,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks,
        model=model,
        track_iterator=True,
    )
    fit.run()


if __name__ == "__main__":
    app.run(main)
