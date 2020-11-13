"""gin wrappers around `tf.keras.Model` methods with tweaks for best-practices."""
from typing import Dict, Iterable, Optional

import gin
import tensorflow as tf
from absl import logging


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
    optimizer = model.optimizer
    optimizer._create_slots(model.trainable_weights)  # pylint:disable=protected-access
    optimizer._create_hypers()  # pylint:disable=protected-access
    optimizer.iterations  # pylint:disable=pointless-statement


@gin.register(module="tf.keras.model")
def compiled(
    model: tf.keras.Model, loss=None, metrics=None, optimizer=None
) -> tf.keras.Model:
    """Mutate model in-place by compiling and return the model for convenience."""
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    return model


@gin.register(module="tf.keras.model")
def fit(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    epochs: int = 1,
    validation_data: Optional[tf.data.Dataset] = None,
    steps_per_epoch: Optional[int] = None,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    initial_epoch: int = 0,
    verbose: bool = True,
) -> tf.keras.callbacks.History:
    """See `tf.keras.Model.fit`."""
    assert_compiled(model)
    if steps_per_epoch is None:
        steps_per_epoch = train_data.cardinality().numpy()
        assert_valid_cardinality(steps_per_epoch)
        train_data = train_data.repeat()
    elif train_data.cardinality() != tf.data.INFINITE_CARDINALITY:
        train_data = train_data.repeat()
    assert train_data.cardinality() == tf.data.INFINITE_CARDINALITY
    assert steps_per_epoch is not None

    if validation_data is not None:
        assert_valid_cardinality(validation_data)
    logging.info("Starting `model.fit`")
    return model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        verbose=verbose,
    )


@gin.register(module="tf.keras.model")
def evaluate(
    model: tf.keras.Model,
    data: tf.data.Dataset,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
) -> Dict[str, float]:
    """See `tf.keras.Model.evaluate`."""
    assert_valid_cardinality(data)
    logging.info("Starting `model.evaluate`")
    return model.evaluate(data, callbacks=callbacks, return_dict=True)


@gin.register(module="tf.keras.model")
def predict(
    model: tf.keras.Model,
    data: tf.data.Dataset,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
):
    """See `tf.keras.Model.predict."""
    assert_valid_cardinality(data)
    logging.info("Starting `model.predict`")
    return model.predict(data, callbacks=callbacks)


def get(identifier) -> tf.keras.Model:
    if isinstance(identifier, tf.keras.Model):
        return identifier

    model = tf.keras.utils.deserialize_keras_object(
        identifier,
        module_objects={
            "Functional": tf.keras.Model,
            "Sequential": tf.keras.Sequential,
        },
    )
    if not isinstance(model, tf.keras.Model):
        raise ValueError(f"Invalid model: {model}")
    return model
