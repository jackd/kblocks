import gin
import tensorflow as tf

from kblocks.framework.batchers import RectBatcher
from kblocks.framework.compilers import compile_classification_model
from kblocks.framework.sources import PipelinedSource, TfdsSource
from kblocks.keras import layers


@gin.configurable()
def simple_cnn(
    inputs_spec,
    num_classes: int,
    conv_filters=(16, 32),
    dense_units=(256,),
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


@gin.configurable()
def cifar100_source(batch_size=16, shuffle_buffer=128):
    def pre_batch_map(image: tf.Tensor, label: tf.Tensor, training: bool):
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        if training:
            image = tf.image.random_flip_left_right(image)
        return image, label

    return PipelinedSource(
        source=TfdsSource("cifar100", split_map={"validation": "test"}),
        batcher=RectBatcher(batch_size),
        pre_batch_map=pre_batch_map,
        shuffle_buffer=shuffle_buffer,
        meta=dict(num_classes=100),
    )


@gin.configurable()
def cifar100_compile(model):
    return compile_classification_model(model, tf.keras.optimizers.Adam())


if __name__ == "__main__":
    from kblocks.framework.trainable import base_trainable, fit

    source = cifar100_source()
    trainable = base_trainable(
        source, simple_cnn, cifar100_compile, "/tmp/kblocks/examples/cifar100"
    )

    fit(trainable, epochs=30)
