import gin
import tensorflow as tf

from kblocks.keras import layers


@gin.register
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


@gin.register
def augment_image_example(
    image: tf.Tensor,
    label: tf.Tensor,
    sample_weight=None,
    noise_stddev=0,
    use_rng: bool = True,
):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    if noise_stddev > 0:
        kwargs = dict(shape=tf.shape(image), stddev=noise_stddev)
        if use_rng:
            noise = tf.random.get_global_generator().normal(**kwargs)
        else:
            noise = tf.random.normal(**kwargs)
        image = image + noise
    return tf.keras.utils.pack_x_y_sample_weight(image, label, sample_weight)
