from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import gin
import tensorflow as tf

from kblocks.data.repeated import RepeatedData, dataset_and_steps


def _validate_data(data: tf.data.Dataset, steps: Optional[int]):
    cardinality = tf.keras.backend.get_value(data.cardinality())
    if cardinality == tf.data.INFINITE_CARDINALITY:
        assert steps is not None
    else:
        assert cardinality > 0
        assert steps is None


@dataclass(frozen=True)
class Trainable:
    """
    Immutable dataclass for storing objects related to model training.

    This is useful for configuration with gin / command line usage. By configuring a
    single `Trainable`, many possible main functions can be run - see
    `kblocks.trainables.mains` - with minimal additional configuration.

    All attributes are intended as used in `tf.keras.Model.fit`, though `callbacks`
    is intended to be used to store only callbacks that influence training
    (e.g. ReduceLROnPlateau) rather than those associated with logging
    (e.g. TensorBoard). Callbacks for logging etc. can be passed to `trainable_fit`.
    """

    model: tf.keras.Model
    train_data: tf.data.Dataset
    steps_per_epoch: Optional[int] = None
    validation_data: Optional[tf.data.Dataset] = None
    validation_steps: Optional[int] = None
    callbacks: Tuple[tf.keras.callbacks.Callback, ...] = ()

    def __post_init__(self):
        assert isinstance(self.model, tf.keras.Model)
        assert isinstance(self.train_data, tf.data.Dataset)
        _validate_data(self.train_data, self.steps_per_epoch)
        if self.validation_data is None:
            assert self.validation_steps is None
        else:
            _validate_data(self.validation_data, self.validation_steps)


@gin.configurable(module="kb.trainables")
def build_trainable(
    model_func: Callable,
    train_data: Union[tf.data.Dataset, RepeatedData],
    steps_per_epoch: Optional[int] = None,
    validation_data: Optional[Union[tf.data.Dataset, RepeatedData]] = None,
    validation_steps: Optional[int] = None,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    compiler: Optional[Callable[[tf.keras.Model], Any]] = None,
) -> Trainable:

    spec = (
        train_data if isinstance(train_data, tf.data.Dataset) else train_data.dataset
    ).element_spec[0]
    model = model_func(spec)
    if compiler is not None:
        compiler(model)
    train_data, steps_per_epoch = dataset_and_steps(train_data, steps_per_epoch)
    validation_data, validation_steps = dataset_and_steps(
        validation_data, validation_steps
    )
    return Trainable(
        model=model,
        train_data=train_data,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=tuple(callbacks),
    )
