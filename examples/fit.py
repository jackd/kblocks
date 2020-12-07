"""
Demonstrate pre-emptible training.

Example usage:
```bash
python fit.py --run=0 --epochs=4
# train in 2 seperate steps
python fit.py --run=1 --epochs=2
python fit.py --run=1 --epochs=4
tensorboard --logdir=/tmp/kblocks/examples/fit
"""
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags

from kblocks.extras.callbacks import ReduceLROnPlateauModule
from kblocks.extras.layers import Dropout
from kblocks.models import fit

os.environ["TF_DETERMINISTIC_OPS"] = "1"


flags.DEFINE_string(
    "root_dir",
    default="/tmp/kblocks/examples/fit",
    help="root save directory for model checkpoints / tb logs",
)
flags.DEFINE_integer("epochs", default=4, help="Total number of epochs to run until.")
flags.DEFINE_integer("seed", default=0, help="Random seed.")
flags.DEFINE_integer("shuffle_buffer", default=1024, help="`shuffle` buffer_size.")
flags.DEFINE_integer("batch_size", default=32, help="Examples per batch.")
flags.DEFINE_integer("run", default=0, help="Models saved to {root_dir}/run-{run:02d}")


def preprocess_example(image, labels):
    """Map functions applied to image classification exampels / batches."""
    image = tf.cast(image, tf.float32) / 255
    return image, labels


def augment_dataset(
    dataset: tf.data.Dataset, noise_stddev: float, seed: int, **map_kwargs
):
    """Apply deterministic augmentations  based on `tf.random.stateless_*` ops."""

    def map_func(example_seed, element):
        image, labels = preprocess_example(*element)
        noise = tf.random.stateless_normal(
            shape=tf.shape(image), seed=example_seed, stddev=noise_stddev
        )
        image += noise
        return image, labels

    random = tf.data.experimental.RandomDataset(seed=seed).batch(2)
    ds = tf.data.Dataset.zip((random, dataset))
    return ds.map(map_func, **map_kwargs)


class BackupAndRestore(tf.keras.callbacks.experimental.BackupAndRestore):
    def on_train_end(self, logs=None):
        # overwrite default behaviour which is to delete final checkpoint
        pass


def main(_):
    FLAGS = flags.FLAGS
    model_dir = f"{FLAGS.root_dir}/run-{FLAGS.run:02d}"
    epochs = FLAGS.epochs
    seed = FLAGS.seed
    shuffle_buffer = FLAGS.shuffle_buffer
    batch_size = FLAGS.batch_size

    tf.random.set_seed(seed)
    tf.random.get_global_generator().reset_from_seed(seed)  # used in Dropout

    # get datasets
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # train
    train_dataset = tfds.load(
        "mnist",
        split="train",
        shuffle_files=True,
        as_supervised=True,
        read_config=tfds.ReadConfig(shuffle_seed=seed),
    )
    steps_per_epoch = len(train_dataset) // batch_size
    train_dataset = (
        train_dataset.repeat().shuffle(shuffle_buffer, seed=seed).batch(batch_size)
    )
    train_dataset = augment_dataset(
        train_dataset,
        noise_stddev=0.1,
        seed=seed,
        num_parallel_calls=AUTOTUNE,
    )
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    # validation
    validation_dataset = tfds.load("mnist", split="test", as_supervised=True)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.map(
        preprocess_example, num_parallel_calls=AUTOTUNE
    )
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)

    inp_spec = train_dataset.element_spec[0]
    inp = tf.keras.Input(shape=inp_spec.shape[1:], dtype=inp_spec.dtype)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inp)
    x = tf.keras.layers.Flatten()(x)
    x = Dropout(0.5)(x)
    logits = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inp, logits)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(),
    )

    callbacks = [
        ReduceLROnPlateauModule(patience=2, factor=0.5),
        tf.keras.callbacks.TensorBoard(model_dir),
        BackupAndRestore(model_dir),
    ]

    fit(
        model,
        train_data=train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        callbacks=callbacks,
        epochs=epochs,
        track_iterator=True,
    )


if __name__ == "__main__":
    app.run(main)
