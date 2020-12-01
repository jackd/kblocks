import tempfile
from typing import Callable, Optional

import gin
import tensorflow as tf
import tqdm
from absl import logging

from kblocks.data.repeated import RepeatedData
from kblocks.models import maybe_prefetch


@gin.configurable(module="kb.profile")
def profile_func(
    func: Callable,
    burn_iters: int = 10,
    run_iters: int = 10,
    path: Optional[str] = None,
    name: str = "profile",
):
    if path is None:
        path = tempfile.mkdtemp()

    for _ in tqdm.trange(burn_iters, desc="Burning in..."):
        func()

    with tf.profiler.experimental.Profile(path):
        for step in tqdm.trange(run_iters, desc="Profiling..."):
            with tf.profiler.experimental.Trace(name, step_num=step):
                func()
    logging.info(
        f"""\
Profile for {run_iters} steps written to {path}

To see results, e.g.
* Start tensorboard: `tensorboard --logdir={path}`
* Navigate to `http://localhost:6006/#profile`
"""
    )


@gin.configurable(module="kb.profile")
def profile_model(
    model: tf.keras.Model,
    data: RepeatedData,
    inference_only: bool = False,
    **kwargs,
):
    dataset = data.dataset
    it = iter(maybe_prefetch(dataset))
    model_func = (
        model.make_predict_function() if inference_only else model.make_train_function()
    )

    def func():
        return model_func(it)

    return profile_func(func, **kwargs, name="predict" if inference_only else "train")
