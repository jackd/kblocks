from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin
import tensorflow as tf
from kblocks.framework.pipelines import ModelPipeline
from kblocks.framework.problems.tfds import TfdsProblem
from kblocks.framework.trainable import Trainable, fit


@gin.configurable()
def simple_cnn(image,
               outputs_spec,
               conv_filters=(16, 32),
               dense_units=(),
               activation='relu'):
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

    num_classes = outputs_spec.shape[-1]
    logits = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs=image, outputs=logits)


def pre_batch_map(image: tf.Tensor, label: tf.Tensor, split=str):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    if split == 'train':
        image = tf.image.random_flip_left_right(image)
    return image, label


@gin.configurable()
def cifar100_problem():
    return TfdsProblem(
        'cifar100',
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        (tf.keras.metrics.SparseCategoricalAccuracy(),),
        split_map={'validation': 'test'},
        pre_batch_map={
            split: functools.partial(pre_batch_map, split=split)
            for split in ('train', 'test')
        },
        outputs_spec=tf.TensorSpec(shape=(None, 100), dtype=tf.float32),
    )


if __name__ == '__main__':
    problem = cifar100_problem()
    pipeline_fn = functools.partial(ModelPipeline, model_fn=simple_cnn)
    trainable = Trainable(problem, pipeline_fn, tf.keras.optimizers.Adam(),
                          '/tmp/kblocks/examples/cifar100')

    fit(trainable, batch_size=16, shuffle_buffer=128, epochs=30)
