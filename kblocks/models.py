"""gin wrappers around `tf.keras.Model` methods with tweaks for best-practices."""
from typing import Optional, Tuple

import gin
import tensorflow as tf


@gin.configurable(module="kb.models")
def compiled(
    model: tf.keras.Model,
    loss=None,
    metrics=None,
    optimizer=None,
    run_eagerly: Optional[bool] = None,
    # steps_per_execution: Optional[int] = None,
) -> tf.keras.Model:
    """Mutate model in-place by compiling and return the model for convenience."""
    model.compile(
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        run_eagerly=run_eagerly,
        # steps_per_execution=steps_per_execution,
    )
    return model


def assert_compiled(model: tf.keras.Model):
    if model.optimizer is None:
        raise RuntimeError("model must be comiled")


def init_optimizer_weights(model: tf.keras.Model):
    """
    Hack to ensure optimizer variables have been created.

    This is normally run on the first optimization step, but various tests save before
    running a single step. Without running this, if the optimizer state is not stored in
    a checkpoint then loading from that checkpoint won't reset the optimizer state to
    default.
    """
    model.optimizer._create_all_weights(  # pylint:disable=protected-access
        model.trainable_weights
    )


def assert_valid_cardinality(
    cardinality: int, allow_unknown: bool = False, allow_infinite: bool = False
):
    """
    Ensure the cardinality (static length) of a dataset satisfies certain requirements.

    Args:
        cardinality: static length of a dataset, e.g. from
            `tf.data.Dataset.cardinality()`
        allow_unknown: if False, `UNKNOWN_CARDINALITY` will raise a `ValueError`.
        allow_infinite: if False, `INFINITE_CARDINALITY` will raise a `ValueError`.

    Raises:
        `ValueError` if conditions are not satisfied.
    """
    if not allow_unknown and cardinality == tf.data.UNKNOWN_CARDINALITY:
        raise ValueError(
            "Unknown cardinality not allowed. If you know the actual finite length, "
            "use `dataset.apply(tf.data.experimental.assert_cardinality("
            "known_cardinality))`"
        )
    if not allow_infinite and cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Infinite cardinality not allowed.")


def as_infinite_iterator(
    dataset: tf.data.Dataset, steps_per_epoch: Optional[int] = None
) -> Tuple[tf.data.Iterator, int]:
    """
    Get an iterator for an infinite dataset and steps_per_epoch.

    Args:
        dataset: possibly infinite dataset.
        steps_per_epoch: number of steps per epoch if `dataset` has infinite
            cardinality, otherwise `None` or `dataset`'s cardinality.

    Returns:
        iterator: tf.data.Iterator of possibly repeated `dataset`.
        steps_per_epoch: number of elements in iterator considered one epoch.

    Raises:
        ValueError is dataset has finite cardinality inconsistent with steps_per_epoch.
    """
    cardinality = tf.keras.backend.get_value(dataset.cardinality())
    if steps_per_epoch is None:
        steps_per_epoch = cardinality
        if cardinality == tf.data.INFINITE_CARDINALITY:
            raise ValueError(
                "steps_per_epoch must be provided if dataset has infinite "
                "cardinality"
            )
        dataset = dataset.repeat()
    elif cardinality != tf.data.INFINITE_CARDINALITY:
        assert cardinality == steps_per_epoch
        dataset = dataset.repeat()
    return iter(dataset), steps_per_epoch


def fit(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    epochs: int = 1,
    steps_per_epoch: Optional[int] = None,
    validation_data: tf.data.Dataset = None,
    validation_steps: Optional[int] = None,
    callbacks: Tuple[tf.keras.callbacks.Callback, ...] = (),
    initial_epoch: int = 0,
    validation_freq: int = 1,
    track_iterator: bool = False,
    verbose: bool = True,
) -> tf.keras.callbacks.History:
    """
    Custom fit implementation.

    Interface is intended to mimic best-practice usage of `tf.keras.Model.fit`.

    Unlike `tf.keras.Model.fit` `_train_iter` is added as an attribute to model. If
    using `tf.train.Checkpoint`s to manage training state, this may result in larger
    files on disk.

    Args:
        model: keras model to train.
        train_data: dataset with (inputs, labels) or (inputs, labels, sample_weights)
        epochs: total number of epochs to train until.
        steps_per_epoch: number of steps per epoch. Must be provided if train_data has
            infinite cardinality.
        validation_data: optional dataset to perform validation on.
        validation_steps: number of steps of validation to perform per epoch.
        callbacks: `tf.keras.callbacks.Callback` instances.
        initial_epoch: starting epoch.
        validation_freq: number of epochs between validation.
        track_iterator: if True, `train_data`'s iterator is added as an attribute to
            `model`, meaning it will be saved in checkpoint's saving `model`.
        verbose: controls verbosity of printed output.

    Returns:
        `tf.keras.callbacks.History` object.

    Raises:
        `AttributeError` if `model` has an existing `_train_iter` attribute and
        `track_iterator` is True.
    """
    train_func = model.make_train_function()
    train_iter, steps_per_epoch = as_infinite_iterator(train_data, steps_per_epoch)
    if hasattr(model, "_train_iter"):
        raise AttributeError("Cannot fit model with existing `_train_iter` attribute.")
    if track_iterator:
        model._train_iter = train_iter  # pylint: disable=protected-access

    cb = tf.keras.callbacks.CallbackList(
        callbacks=callbacks, add_history=True, add_progbar=verbose, model=model
    )
    cb.set_params(dict(epochs=epochs, verbose=int(verbose), steps=steps_per_epoch))

    cb.on_train_begin()
    initial_epoch = (
        model._maybe_load_initial_epoch_from_ckpt(  # pylint: disable=protected-access
            initial_epoch
        )
    )

    training_logs = None
    model.stop_training = False
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        cb.on_epoch_begin(epoch)

        logs = None
        for step in range(steps_per_epoch):
            cb.on_train_batch_begin(step)
            logs = train_func(train_iter)
            cb.on_train_batch_end(step, logs)
            if model.stop_training:
                break
        assert logs is not None
        epoch_logs = logs
        if (
            validation_data is not None
            and model._should_eval(  # pylint: disable=protected-access
                epoch, validation_freq
            )
        ):
            logs = model.evaluate(
                validation_data,
                steps=validation_steps,
                callbacks=cb,
                return_dict=True,
            )
            epoch_logs.update({"val_" + name: val for name, val in logs.items()})
        cb.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        if model.stop_training:
            break
    cb.on_train_end(logs=training_logs)
    if track_iterator:
        del model._train_iter
    return model.history


def get(identifier) -> tf.keras.Model:
    if isinstance(identifier, tf.keras.Model):
        return identifier

    model = tf.keras.utils.deserialize_keras_object(
        identifier,
        module_objects={
            "Functional": tf.keras.Model,
            "Sequential": tf.keras.Sequential,
        },
        printable_module_name="Model",
    )
    if not isinstance(model, tf.keras.Model):
        raise ValueError(f"Invalid model: {model}")
    return model
