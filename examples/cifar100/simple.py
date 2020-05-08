import gin
import tensorflow as tf
from kblocks.framework.sources import TfdsSource
from kblocks.framework.sources import PipelinedSource
from kblocks.framework.pipelines import BasePipeline
from kblocks.framework.trainable import base_trainable, fit
from kblocks.framework.compilers import compile_classification_model


@gin.configurable()
def simple_cnn(
    inputs_spec,
    num_classes: int,
    conv_filters=(16, 32),
    dense_units=(),
    activation="relu",
):
    image = tf.keras.Input(shape=inputs_spec.shape[1:], dtype=inputs_spec.dtype)
    x = image
    for f in conv_filters:
        x = tf.keras.layers.Conv2D(f, 3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = tf.keras.layers.Dense(u)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

    logits = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs=image, outputs=logits)


def pre_batch_map(image: tf.Tensor, label: tf.Tensor):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    learning_phase = tf.keras.backend.learning_phase()
    assert isinstance(learning_phase, bool)
    if learning_phase:
        image = tf.image.random_flip_left_right(image)
    return image, label


@gin.configurable()
def cifar100_source(batch_size=16, shuffle_buffer=128):
    pipeline = BasePipeline(
        batch_size=batch_size,
        pre_batch_map=pre_batch_map,
        shuffle_buffer=shuffle_buffer,
    )
    base_source = TfdsSource("cifar100", split_map={"validation": "test"})
    source = PipelinedSource(source=base_source, pipeline=pipeline)
    return source


@gin.configurable()
def cifar100_compile(model):
    return compile_classification_model(model, tf.keras.optimizers.Adam())


if __name__ == "__main__":
    source = cifar100_source()
    trainable = base_trainable(
        source, simple_cnn, cifar100_compile, "/tmp/kblocks/examples/cifar100"
    )

    fit(trainable, epochs=30)
