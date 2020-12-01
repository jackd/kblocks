"""gin wrappers around `tf.keras.Model` methods with tweaks for best-practices."""
from typing import Iterable, Optional

import gin
import tensorflow as tf

from kblocks.data.repeated import RepeatedData


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
    optimizer = model.optimizer
    optimizer._create_slots(model.trainable_weights)  # pylint:disable=protected-access
    optimizer._create_hypers()  # pylint:disable=protected-access
    optimizer.iterations  # pylint:disable=pointless-statement


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


def maybe_prefetch(
    dataset: tf.data.Dataset, buffer_size: int = tf.data.experimental.AUTOTUNE
):
    if dataset.__class__.__name__ != "PrefetchDataset":
        dataset = dataset.prefetch(buffer_size)
    return dataset


# def fit(
#     model: tf.keras.Model,
#     train_data: RepeatedData,
#     epochs: int = 1,
#     validation_data: Optional[RepeatedData] = None,
#     callbacks: Iterable[tf.keras.callbacks.Callback] = (),
#     initial_epoch: int = 0,
#     validation_freq: int = 1,
#     verbose: bool = True,
#     use_iterators: bool = False,
# ) -> tf.keras.callbacks.History:
#     assert_compiled(model)
#     model.stop_training = False
#     train_func = model.make_train_function()
#     validation_func = model.make_test_function()
#     train_iter = iter(train_data.dataset)
#     if validation_data is not None:
#         validation_iter = iter(validation_data.dataset)

#     if use_iterators:
#         assert not hasattr(model, "train_iter")
#         model.train_iter = train_iter
#     cb = tf.keras.callbacks.CallbackList(
#         callbacks=callbacks, add_history=True, add_progbar=verbose, model=model
#     )

#     cb.on_train_begin()

#     initial_epoch = model._maybe_load_initial_epoch_from_ckpt(  # pylint: disable=protected-access
#         initial_epoch
#     )
#     for epoch in range(initial_epoch, epochs):
#         model.reset_metrics()
#         cb.on_epoch_begin(epoch)

#         logs = None
#         for step in range(train_data.steps_per_epoch):
#             cb.on_train_batch_begin(step)
#             logs = train_func(train_iter)
#             cb.on_train_batch_end(step, logs)
#             if model.stop_training:
#                 break
#         assert logs is not None
#         epoch_logs = logs
#         # validation
#         if (
#             validation_data is not None
#             and model._should_eval(  # pylint: disable=protected-access
#                 epoch, validation_freq
#             )
#         ):
#             cb.on_test_begin()
#             model.reset_metrics()
#             for step in range(validation_data.steps_per_epoch):
#                 cb.on_test_batch_begin(step)
#                 logs = validation_func(validation_iter)
#                 cb.on_test_batch_end(step, logs)
#             cb.on_test_end(logs)
#             epoch_logs.update({"val_" + name: val for name, val in logs.items()})
#         cb.on_epoch_end(epoch, epoch_logs)
#         training_logs = epoch_logs
#         if model.stop_training:
#             break
#     cb.on_train_end(logs=training_logs)
#     if use_iterators:
#         del model.train_iter
#     return model.history


def fit(
    model: tf.keras.Model,
    train_data: RepeatedData,
    epochs: int = 1,
    validation_data: Optional[RepeatedData] = None,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    initial_epoch: int = 0,
    validation_freq: int = 1,
    verbose: bool = True,
    use_iterators: bool = False,
) -> tf.keras.callbacks.History:
    """
    Wrapper around `tf.keras.Model.fit` that enforces best practices.

    The following modifications are made.

    1. It uses `RepeatedData` rather than standard datasets. For pipelines with
        augmentation, these are generally easier to to construct and return from a
        function separate from other arguments. Accepting one RepeatedData (rather than
        e.g. validaiton_data, validation_steps) makes configuration easier with gin.
    2. It adds train data's iterator as an attribute onto model, meaning checkpoints
        (e.g. those created by `tf.keras.callbacks.experimental.BackupAndRestore`) that
        save `model` also save the iterator state.
    3. If either RepeatedData datasets are not `PrefetchDataset`s a prefetch(AUTOTUNE)
        transform is added.

    See `Fit` for an experiment wrapper around this method.
    """
    if use_iterators:
        raise Exception("Iterator checkpointing currently not supported in TF.")
    assert_compiled(model)

    train_iter = maybe_prefetch(train_data.dataset)
    if use_iterators:
        train_iter = iter(train_iter)
        assert not hasattr(model, "train_iter")
        model.train_iter = train_iter

    train_steps = train_data.steps_per_epoch
    # We add `train_iter` as an attribute to model so it can be saved along with model

    # possible memory leak when using validation dataset rather than iterator??
    if validation_data is None:
        validation_iter = None
        validation_steps = None
    else:
        validation_iter = maybe_prefetch(validation_data.dataset)
        if use_iterators:
            validation_iter = iter(validation_iter)
        validation_steps = validation_data.steps_per_epoch

    history = model.fit(
        train_iter,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_iter,
        validation_steps=validation_steps,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        validation_freq=validation_freq,
        verbose=verbose,
    )
    if use_iterators:
        del model.train_iter
    return history


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
