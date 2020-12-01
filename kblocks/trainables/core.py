from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import gin
import tensorflow as tf

from kblocks.data.repeated import RepeatedData, repeated_data


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
    train_data: RepeatedData
    validation_data: Optional[RepeatedData] = None
    callbacks: Tuple[tf.keras.callbacks.Callback, ...] = ()


@gin.configurable(module="kb.trainables")
def build_trainable(
    model_func: Callable,
    train_data: Union[tf.data.Dataset, RepeatedData],
    validation_data: Optional[Union[tf.data.Dataset, RepeatedData]] = None,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    compiler: Optional[Callable[[tf.keras.Model], Any]] = None,
) -> Trainable:
    if isinstance(train_data, tf.data.Dataset):
        train_data = RepeatedData(train_data)
    if isinstance(validation_data, tf.data.Dataset):
        validation_data = RepeatedData(validation_data)
    model = model_func(train_data.dataset.element_spec[0])
    if compiler is not None:
        compiler(model)
    return Trainable(
        model=model,
        train_data=repeated_data(train_data),
        validation_data=repeated_data(validation_data),
        callbacks=callbacks,
    )
