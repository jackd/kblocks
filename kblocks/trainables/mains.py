"""main function implementations."""
from typing import Tuple

import gin
import numpy as np
import tensorflow as tf
import tqdm
from absl import logging

from kblocks import benchmarks as bm
from kblocks.experiments.fit import Fit
from kblocks.models import as_infinite_iterator, fit
from kblocks.profile import profile_model
from kblocks.trainables.core import Trainable


@gin.configurable(module="kb.trainables")
def trainable_fit(trainable: Trainable, **kwargs) -> Fit:
    callbacks = trainable.callbacks + tuple(kwargs.pop("callbacks", ()))
    return Fit(
        model=trainable.model,
        train_data=trainable.train_data,
        steps_per_epoch=trainable.steps_per_epoch,
        validation_data=trainable.validation_data,
        validation_steps=trainable.validation_steps,
        callbacks=callbacks,
        **kwargs,
    )


@gin.configurable(module="kb.trainables")
def get_dataset(trainable: Trainable, training: bool = True) -> tf.data.Dataset:
    return trainable.train_data if training else trainable.validation_data


@gin.configurable(module="kb.trainables")
def benchmark_trainable_model(trainable: Trainable, training: bool = True, **kwargs):
    """Run benchmark on model training/inference. Does not include callbacks."""
    return bm.benchmark_model(
        trainable.model,
        get_dataset(trainable, training),
        **kwargs,
    )


@gin.configurable(module="kb.trainables")
def benchmark_trainable_data(trainable: Trainable, training: bool = True, **kwargs):
    """Run benchmark on trainable data."""
    return bm.benchmark_dataset(get_dataset(trainable, training), **kwargs)


@gin.configurable(module="kb.trainables")
def profile_trainable(trainable: Trainable, training: bool = True, **kwargs):
    # skips callbacks
    return profile_model(trainable.model, get_dataset(trainable, training), **kwargs)


@gin.configurable(module="kb.trainables")
def check_weight_updates(trainable: Trainable, epochs: int = 1) -> Tuple[int, int]:
    """Fit for an epoch and log how many trainable_weights are unchanged."""
    model = trainable.model
    weights = model.trainable_weights
    original_weights = [tf.keras.backend.get_value(w) for w in weights]
    fit(
        trainable.model,
        trainable.train_data,
        epochs=epochs,
        callbacks=trainable.callbacks,
    )
    trained_weights = [tf.keras.backend.get_value(w) for w in weights]
    total_elements = 0
    total_same = 0
    for (original, trained, variable) in zip(
        original_weights, trained_weights, weights
    ):
        same = np.count_nonzero(original == trained)
        num_elements = variable.shape.num_elements()
        if same != 0:
            total_same += same
            logging.info(f"{variable.name}: {same} of {num_elements} the same")
        total_elements += num_elements
    if total_same == 0:
        logging.info("All trainable weights different!")
    else:
        logging.info(f"Total: {total_same} of {total_elements} the same")
    return total_same, total_elements


@gin.configurable(module="kb.trainables")
def iterate_over_data(trainable: Trainable, training: bool = True, epochs: int = -1):
    if training:
        it, steps_per_epoch = as_infinite_iterator(
            trainable.train_data, trainable.steps_per_epoch
        )
    else:
        it, steps_per_epoch = as_infinite_iterator(
            trainable.validation_data, trainable.validation_steps
        )
    epoch = 0
    while True:
        suffix = f"{epoch + 1} / {epochs}" if epochs > 0 else str(epoch + 1)
        for _ in tqdm.trange(steps_per_epoch, desc=f"Iterating over epoch {suffix}..."):
            it.get_next()
        epoch += 1
        if epoch == epochs:
            break
