import functools
import os
from typing import Callable

import gin
import tensorflow as tf

from kblocks.framework.sources import DataSource
from kblocks.keras.tuner import Hyperband


@gin.configurable(module="kb.framework")
def tune(
    model_fn: Callable,
    source: DataSource,
    objective,
    max_epochs: int,
    tuner_fn=Hyperband,
    directory=None,
    project_name=None,
):
    tuner = tuner_fn(
        functools.partial(model_fn, **source.meta),
        objective,
        max_epochs,
        directory=directory,
        project_name=project_name,
    )
    splits = ("train", "validation")
    train_ds, val_ds = (source.get_dataset(split) for split in splits)
    train_steps, val_steps = (source.examples_per_epoch(split) for split in splits)
    tuner.search(
        x=train_ds,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[
            tf.keras.callbacks.TensorBoard(os.path.join(directory, "summaries"))
        ],
    )
    tuner.search_space_summary()
    return tuner
