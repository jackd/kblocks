from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable, Optional, Tuple
import tensorflow as tf


def custom_fit(
    model,
    x: tf.data.Dataset,
    epochs: int = 1,
    verbose: int = 1,
    callbacks: Iterable[tf.keras.callbacks.Callback] = [],
    validation_data: Optional[tf.data.Dataset] = None,
    initial_epoch: int = 0,
    steps_per_epoch: Optional[int] = None,
    validation_steps: Optional[int] = None,
    validation_freq: int = 1,
    loss=None,
    metrics=None,
    optimizer=None,
):
    if not tf.executing_eagerly():
        raise RuntimeError("custom_fit should be called in eager mode")

    if loss is None:
        loss = model.loss

    if metrics is None:
        metrics = model.metrics

    if optimizer is None:
        optimizer = model.optimizer

    history = tf.keras.callbacks.History()
    prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
    prog_bar.set_params(
        dict(
            verbose=verbose,
            epochs=epochs,
            metrics=("loss",) + tuple(m.name for m in metrics),
            steps=steps_per_epoch,
        )
    )
    callbacks = [history, prog_bar,] + list(callbacks)

    def get_total_loss(labels, preds, weights=None):
        loss_val = loss(labels, preds, weights)
        model_losses = list(model.losses)
        if len(model_losses) > 0:
            model_losses.append(loss_val)
            loss_val = tf.add_n(model_losses)
        return loss_val

    def train_step(features, labels, weights=None):
        with tf.GradientTape() as tape:
            preds = model(features)
            loss_val = get_total_loss(labels, preds, weights)

        gradients = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for m in metrics:
            m.update_state(labels, preds, weights)
        return loss_val

    def val_step(features, labels, weights=None):
        preds = model(features)
        loss_val = get_total_loss(labels, preds, weights)
        for m in metrics:
            m.update_state(labels, preds, weights)
        return loss_val

    # don't annotate so pyright shuts up
    train_step = tf.function(train_step)
    val_step = tf.function(val_step)

    def reset_metric_states():
        for m in metrics:
            m.reset_states()

    logs = {}
    for c in callbacks:
        c.on_train_begin(logs)

    if steps_per_epoch is not None:
        x = x.take(steps_per_epoch)

    if validation_steps is not None and validation_data is not None:
        validation_data = validation_data.take(validation_steps)

    for epoch in range(initial_epoch, epochs):
        reset_metric_states()
        logs = {}
        for c in callbacks:
            c.on_epoch_begin(epoch, logs)
        for batch, args in enumerate(x):
            for c in callbacks:
                c.on_train_batch_begin(batch, logs)
            loss_val = train_step(*args)
            logs = {m.name: m.result().numpy() for m in metrics}
            logs["loss"] = loss_val.numpy()
            for c in callbacks:
                c.on_train_batch_end(batch, logs)

        logs = {m.name: m.result().numpy() for m in metrics}

        if epoch % validation_freq == 0 and validation_data is not None:
            reset_metric_states()
            for args in validation_data:
                val_step(*args)

            # add validation metrics
            for m in metrics:
                logs["val_{}".format(m.name)] = m.result().numpy()

        for c in callbacks:
            c.on_epoch_end(epoch, logs)
    return history
