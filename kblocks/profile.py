import tempfile
from typing import Callable, Optional

import gin
import tensorflow as tf
import tqdm

from kblocks.trainables import Trainable

# @tf.function
# def train_step(model, inputs, labels, sample_weight=None):
#     variables = model.trainable_variables
#     with tf.GradientTape() as tape:
#         prediction = model(inputs)
#         loss = model.loss(labels, prediction, sample_weight)
#     grads = tape.gradient(loss, variables)
#     model.optimizer.apply_gradients(zip(grads, variables))


# @tf.function
# def inference_step(model, inputs, labels=None, sample_weight=None):
#     return model(inputs)


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
    print(
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
    dataset: tf.data.Dataset,
    inference_only: bool = False,
    **kwargs,
):
    it = iter(dataset)
    # model_func = inference_step if inference_only else train_step
    model_func = (
        model.make_predict_function() if inference_only else model.make_train_function()
    )

    def func():
        return model_func(it)
        # return model_func(model, *next(it))

    return profile_func(func, **kwargs, name="predict" if inference_only else "train")


@gin.configurable(module="kb.profile")
def profile_trainable(trainable: Trainable, train_split: bool = True, **kwargs):
    source = trainable.train_source if train_split else trainable.validation_source
    return profile_model(trainable.model, source.dataset, **kwargs)
