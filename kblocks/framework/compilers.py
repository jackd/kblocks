from typing import Optional

import gin
import tensorflow as tf


@gin.configurable(module="kb.framework")
def compile_model(
    model: tf.keras.Model,
    loss=None,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    metrics=None,
    loss_weights=None,
):
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights
    )


@gin.configurable(module="kb.framework")
def compile_classification_model(
    model: tf.keras.Model,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    from_logits: bool = True,
):
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


@gin.configurable(module="kb.framework")
def compile_binary_classification_model(
    model: tf.keras.Model,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    from_logits: bool = True,
):
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=from_logits),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
