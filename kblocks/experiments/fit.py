import os
from typing import Iterable, Optional, Tuple, Union

import gin
import tensorflow as tf

import kblocks.extras.callbacks as ecb
import kblocks.keras.callbacks as kcb
from kblocks.data.repeated import RepeatedData, repeated_data
from kblocks.experiments.core import Experiment
from kblocks.models import fit
from kblocks.path import expand


@gin.configurable(module="kb.experiments")
def model_callbacks(
    *,
    terminate_on_nan: bool = True,
    learning_rate_scheduler: bool = False,
    reduce_lr_on_plateau: bool = False,
    reduce_lr_on_plateau_module: bool = False,
    early_stopping: bool = False,
    early_stopping_module: bool = False,
) -> Tuple[tf.keras.callbacks.Callback, ...]:
    """
    Get callbacks that influence model training.

    Callbacks from from `kblocks.[keras,extras].callbacks`.

    Args:
        terminate_on_nan: include `tf.keras.callbacks.TerminateOnNaN`.
        learning_rate_scheduler: include
            `kblocks.keras.callbacks.LearningRateScheduler`.
        reduce_lr_on_plateau: include `kblocks.keras.callbacks.ReduceLROnPlateau`.
            At most one of `reduce_lr_on_plateau` and `reduce_lr_on_plateau_module` can
            be True.
        reduce_lr_on_plateau_module: include
            `kblocks.extras.callbacks.ReduceLROnPlateauModule`. At most one of
            `reduce_lr_on_plateau` and `reduce_lr_on_plateau_module` can be True.
        early_stopping: include `kblocks.keras.callbacks.EarlyStopping`. At most one of
            `early_stopping` and `early_stopping_module` can be True.
        early_stopping_module: include `kblocks.extras.callbacks.EarlyStoppingModule`.
            At most one of `early_stopping` and `early_stopping_module` can be True.

    Returns:
        List of callbacks.
    """
    if early_stopping and early_stopping_module:
        raise ValueError(
            "At most one of `early_stopping` and `early_stopping_module` can be True."
        )
    if reduce_lr_on_plateau and reduce_lr_on_plateau_module:
        raise ValueError(
            "At most one of `reduce_lr_on_plateau` and `reduce_lr_on_plateau_module` "
            "can be True"
        )

    callbacks = []
    if terminate_on_nan:
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if reduce_lr_on_plateau:
        callbacks.append(kcb.ReduceLROnPlateau())
    if early_stopping:
        callbacks.append(kcb.EarlyStopping())
    if reduce_lr_on_plateau_module:
        callbacks.append(ecb.ReduceLROnPlateauModule())
    if early_stopping_module:
        callbacks.append(ecb.EarlyStoppingModule())
    if learning_rate_scheduler:
        callbacks.append(kcb.LearningRateScheduler())
    return tuple(callbacks)


@gin.configurable(module="kb.experiments")
def logging_callbacks(
    experiment_dir: str,
    *,
    tb: bool = True,
    absl: bool = True,
    csv: bool = False,  # doesn't work with validation_freq != 1 and backup...
    backup: bool = True,
) -> Tuple[tf.keras.callbacks.Callback, ...]:
    """
    Get callbacks configured with gin from `kblocks.[extras,keras].callbacks`.

    Each `true` argument corresponds to a callback from `kblocks.extras.callbacks`,
    `kblocks.keras.callbacks`, or `tf.keras.callbacks`. Path arguments are passed to
    constructors based on subpaths of `experiment_dir`. Unless mentioned below, other
    required arguments are expected to be configured via `gin`.

    Args:
        experiment_dir: root directory in which to save relevant files.
        tb: include `kblocks.keras.callbacks.TensorBoard`.
        absl: include `kblocks.extras.callbacks.AbslLogger`.
        csv: include `kblocks.keras.callbacks.CSVLogger`.
        backup: include `kblocks.keras.callbacks.BackupAndRestore`.

    Returns:
        Tuple of `tf.keras.callbacks.Callback`s.

    Raises:
        `ValueError` if `early_stopping and early_stopping_module` or
                        `reduce_lr_on_plateau and reduce_lr_on_plateau_module`.
    """
    experiment_dir = expand(experiment_dir)
    callbacks = []
    if backup:
        callbacks.append(kcb.BackupAndRestore(os.path.join(experiment_dir, "backup")))
    if tb:
        callbacks.append(kcb.TensorBoard(os.path.join(experiment_dir, "tb")))
    if absl:
        callbacks.append(ecb.AbslLogger())
    if csv:
        callbacks.append(
            kcb.CSVLogger(os.path.join(experiment_dir, "log.csv"), append=backup)
        )

    return tuple(callbacks)


@gin.configurable(module="kb.experiments")
def fit_callbacks():
    """Get `model_callbacks` and `logging_callbacks`."""
    # pylint: disable=no-value-for-parameter
    return model_callbacks() + logging_callbacks()
    # pylint: enable=no-value-for-parameter


@gin.configurable(module="kb.experiments")
class Fit(Experiment):
    """
    Experiment wrapper around `fit`.

    Args:
        experiment_dir: directory to save status / operative config.
        model: `tf.keras.Model` or a callable that maps `input_spec` to a
            `tf.keras.Model`, where `input_spec == train_data.element_spec[0]`
        train_data: dataset with (x, y, sample_weight?)
        **kwargs: see `tf.keras.Model.fit`.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        train_data: Union[tf.data.Dataset, RepeatedData],
        epochs: int = 1,
        validation_data: Optional[Union[tf.data.Dataset, RepeatedData]] = None,
        callbacks: Iterable[tf.keras.callbacks.Callback] = (),
        validation_freq: int = 1,
        verbose: bool = True,
        use_iterators: bool = False,
        **kwargs,
    ):
        if not isinstance(model, tf.keras.Model):
            model = model(train_data.element_spec[0])
            if not isinstance(model, tf.keras.Model):
                raise ValueError(
                    "model must be a `tf.keras.Model` or callable mapping "
                    f"input spec(s) to a Model, got {model}"
                )
        self._model = model
        self._train_data = repeated_data(train_data)
        self._epochs = epochs
        self._validation_data = repeated_data(validation_data)
        self._callbacks = tuple(callbacks)
        self._validation_freq = validation_freq
        self._verbose = verbose
        self._use_iterators = use_iterators
        super().__init__(**kwargs)

    def _run(self, start_status):
        return fit(
            model=self._model,
            train_data=self._train_data,
            epochs=self._epochs,
            validation_data=self._validation_data,
            callbacks=self._callbacks,
            validation_freq=self._validation_freq,
            verbose=self._verbose,
            use_iterators=self._use_iterators,
        )
