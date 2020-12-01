from typing import Callable, Optional

import gin
import tensorflow as tf
import tfrng

from kblocks.keras import layers
from kblocks.models import RepeatedData


@gin.configurable
def simple_cnn(
    inputs_spec,
    num_classes: int,
    conv_filters=(16, 32),
    dense_units=(256, 256, 256),
    activation="relu",
):
    image = tf.keras.Input(shape=inputs_spec.shape[1:], dtype=inputs_spec.dtype)
    x = image
    for f in conv_filters:
        x = layers.Conv2D(f, 3)(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = layers.Dense(u)(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

    logits = layers.Dense(num_classes, activation=None)(x)

    return tf.keras.Model(inputs=image, outputs=logits)


@gin.configurable
def augment_image_example(
    image: tf.Tensor, label: tf.Tensor, sample_weight=None, noise_stddev=0
):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    if noise_stddev > 0:
        image = image + tfrng.normal(shape=tf.shape(image), stddev=noise_stddev)
    return tf.keras.utils.pack_x_y_sample_weight(image, label, sample_weight)


@gin.configurable
def get_augmented_data(
    dataset: tf.data.Dataset,
    batch_size: int,
    map_func: Callable,
    shuffle_buffer: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    augment_seed: Optional[int] = None,
    use_stateless_map: bool = False,
) -> RepeatedData:
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer, seed=shuffle_seed)
    dataset = dataset.batch(batch_size)
    steps_per_epoch = tf.keras.backend.get_value(dataset.cardinality())
    # repeat before map so stateless map is different across epochs
    dataset = dataset.repeat()
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if use_stateless_map:
        dataset = dataset.apply(
            tfrng.data.stateless_map(
                map_func,
                seed=augment_seed,
                num_parallel_calls=AUTOTUNE,
            )
        )
    else:
        # if map_func has random elements this won't be deterministic
        dataset = dataset.map(map_func, num_parallel_calls=AUTOTUNE)
    return RepeatedData(dataset, steps_per_epoch)
