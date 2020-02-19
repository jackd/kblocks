from typing import Optional
import tensorflow as tf
import gin


@gin.configurable(module='kb.framework')
def compile_model(model: tf.keras.Model,
                  loss=None,
                  optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                  metrics=None,
                  loss_weights=None):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  loss_weights=loss_weights)


@gin.configurable(module='kb.framework')
def compile_classification_model(
        model: tf.keras.Model,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None):
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
