import os
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import gin
import numpy as np
import tensorflow as tf
import tqdm
from absl import logging

from kblocks import utils
from kblocks.benchmark_utils import summarize
from kblocks.extras import callbacks as cb
from kblocks.framework.sources import DataSource


def _flatten_dataset_features(dataset):
    def map_fn(inputs, labels, weights=None):
        inputs = tuple(tf.nest.flatten(inputs))
        return (inputs, labels) if weights is None else (inputs, labels, weights)

    return dataset.map(map_fn)


def _updated(base: Optional[Mapping], **kwargs):
    if base is None:
        return kwargs
    base = dict(**base)
    base.update(kwargs)
    return base


@gin.configurable(module="kb.framework")
def base_trainable(
    source: DataSource,
    model_fn: Callable,
    compiler: Optional[Callable[[tf.keras.Model], Any]],
    model_dir: Optional[str] = None,
):
    model = model_fn(source.element_spec[0], **source.meta)
    if compiler is not None:
        compiler(model)
    return Trainable(source, model, model_dir)


@gin.configurable(module="kb.framework")
class Trainable:
    def __init__(
        self, source: DataSource, model: tf.keras.Model, model_dir: Optional[str] = None
    ):
        if model_dir is not None:
            model_dir = os.path.expanduser(os.path.expandvars(model_dir))
        self._model_dir = model_dir
        self._source = source
        self._model = model

    @property
    def model_dir(self) -> Optional[str]:
        return self._model_dir

    @property
    def source(self) -> DataSource:
        return self._source

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    def check_weight_updates(self, total_steps):
        model = self.model
        weight_vals = [w.numpy() for w in model.trainable_weights]
        ds = self._source.get_dataset("train").take(total_steps)

        model.fit(ds, steps_per_epoch=total_steps, epochs=1)
        for weight, orig in zip(model.trainable_weights, weight_vals):
            if np.allclose(weight.numpy(), orig):
                logging.info(
                    "{} ({}) value largely unchanged".format(
                        weight.name, tuple(weight.shape)
                    )
                )

    def check_gradients(self, burn_iters=50, run_iters=5):
        model = self.model
        loss = model.loss
        dataset = self._source.get_dataset("train")
        weights = model.trainable_weights

        def compute_grads_and_vars(args):
            if len(args) == 3:
                features, labels, label_weights = args
            else:
                features, labels = args
                label_weights = None
            with tf.GradientTape() as tape:
                inference = model(features)
                loss_val = loss(labels, inference, label_weights)
                model_losses = list(model.losses)
                if len(model_losses) > 0:
                    model_losses.append(loss_val)
                    loss_val = tf.add_n(model_losses)
                grads = tape.gradient(loss_val, weights)
            return zip(grads, weights)

        for args in tqdm.tqdm(
            dataset.take(burn_iters), total=burn_iters, desc="Burning in..."
        ):
            grads_and_vars = compute_grads_and_vars(args)
            model.optimizer.apply_gradients(grads_and_vars)

        for i, args in enumerate(dataset.take(run_iters)):
            logging.info("Starting run step {}".format(i))
            grads_and_vars = compute_grads_and_vars(args)
            for grad, weight in grads_and_vars:
                if grad is None:
                    logging.info(
                        "Gradient for weight {} ({}) is None".format(
                            weight.name, tuple(weight.shape)
                        )
                    )
                elif np.allclose(grad.numpy(), 0):
                    logging.info(
                        f"Gradients for weight {weight.name} ({tuple(weight.shape)}) "
                        f"are all close to zero, largest is "
                        f"{tf.reduce_max(tf.abs(grad)).numpy()}"
                    )
        logging.info("Finished checking gradients")

    def benchmark(
        self,
        burn_iters: int,
        min_iters: int,
        dataset_only: bool = False,
        forward_only: bool = False,
        fixed_inputs: bool = False,
    ):
        kwargs = dict(burn_iters=burn_iters, min_iters=min_iters)
        if dataset_only:
            return self.benchmark_dataset(**kwargs)
        return self.benchmark_pipeline(
            forward_only=forward_only, fixed_inputs=fixed_inputs, **kwargs
        )

    def benchmark_dataset(self, burn_iters: int, min_iters: int):
        dataset = self._source.get_dataset("train")
        op = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        bm = tf.test.Benchmark()
        with tf.compat.v1.Session() as sess:
            logging.info("Starting benchmarking...")
            result = bm.run_op_benchmark(
                sess, op, burn_iters=burn_iters, min_iters=min_iters
            )
            summarize(result)
        return result

    def benchmark_pipeline(
        self, burn_iters: int, min_iters: int, forward_only=False, fixed_inputs=False
    ):
        dataset = self._source.get_dataset("train")

        inputs, labels = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

        if fixed_inputs:
            logging.info("Getting hack inputs...")
            with tf.compat.v1.Session() as sess:
                inputs, labels = sess.run((inputs, labels))
            logging.info("Got inputs!")

            inputs, labels = tf.nest.map_structure(
                lambda x: tf.constant(x)
                if isinstance(x, np.ndarray)
                else tf.RaggedTensor.from_nested_row_splits(
                    tf.constant(x.flat_values),
                    [tf.constant(i) for i in x.nested_row_splits],
                ),
                (inputs, labels),
            )

        model = self.model
        out = model(inputs)

        if forward_only:
            train_op = out
        else:
            loss = model.loss(labels, out)
            optimizer = model.optimizer
            weights = model.trainable_weights
            grads = optimizer.get_gradients(loss, weights)
            grads_and_vars = tuple(
                (g, v) for g, v in zip(grads, weights) if g is not None
            )
            train_op = optimizer.apply_gradients(grads_and_vars)

        bm = tf.test.Benchmark()
        with tf.compat.v1.Session() as sess:
            logging.info("Initializing variables...")

            sess.run(tf.compat.v1.global_variables_initializer())

            logging.info("Starting benchmarking...")
            result = bm.run_op_benchmark(
                sess, train_op, burn_iters=burn_iters, min_iters=min_iters
            )
        summarize(result)
        return result

    def evaluate(self, chkpt_kwargs: Optional[Mapping[str, Any]] = None):
        model_dir = self.model_dir
        model = self.model
        source = self.source
        split = "validation"

        ds = source.get_dataset(split)
        steps = source.epoch_length(split)

        if model_dir is None:
            logging.warning("No model_dir provided - evaluating without restoration")
        else:
            chkpt_dir = os.path.join(model_dir, "chkpts")
            chkpt_callback = cb.CheckpointCallback(
                **_updated(chkpt_kwargs, directory=chkpt_dir)
            )

            chkpt_callback.set_model(model)
            chkpt = chkpt_callback.checkpoint()
            if chkpt is None:
                logging.warning("No checkpoint found - evaluating without restoration")
            else:
                initial_epoch = chkpt_callback.epoch(chkpt)
                chkpt_callback.restore(initial_epoch).expect_partial()
        model.evaluate(ds, steps=steps)

    def fit(
        self,
        epochs: int,
        verbose: bool = True,
        callbacks: Sequence[tf.keras.callbacks.Callback] = (),
        chkpt_kwargs: Optional[Mapping[str, Any]] = None,
        validation_freq: int = 1,
        profile_batch: Union[int, str, Tuple[int, int]] = 2,
        build_only: bool = False,
    ):
        source = self.source
        model = self.model
        model_dir = self.model_dir

        splits = ("train", "validation")
        train_ds, val_ds = (source.get_dataset(s) for s in splits)
        spec = train_ds.element_spec[0]
        flat_spec = tuple(tf.nest.flatten(train_ds))
        if flat_spec != spec:
            train_ds, val_ds = (
                _flatten_dataset_features(ds) for ds in (train_ds, val_ds)
            )
        train_steps, val_steps = (source.epoch_length(s) for s in splits)

        used_callbacks = [
            cb.AbslLogger(),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        utils.init_optimizer_weights(model)

        # do_fit needs to be in a session if running in graph mode
        def do_fit():
            if model_dir is not None:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)
                chkpt_dir = os.path.join(model_dir, "chkpts")
                chkpt_callback = cb.CheckpointCallback(
                    **_updated(chkpt_kwargs, directory=chkpt_dir)
                )

                chkpt_callback.set_model(model)
                chkpt = chkpt_callback.checkpoint()
                if chkpt is None:
                    initial_epoch = 0
                else:
                    initial_epoch = chkpt_callback.epoch(chkpt)
                    chkpt_callback.restore(initial_epoch).assert_consumed()

                used_callbacks.append(chkpt_callback)
            else:
                initial_epoch = 0

            if initial_epoch == 0 and not tf.executing_eagerly():
                tf.compat.v1.get_default_session().run(
                    tf.compat.v1.global_variables_initializer()
                )

            used_callbacks.extend(callbacks)
            if model_dir is not None:
                used_callbacks.extend(
                    [
                        tf.keras.callbacks.TensorBoard(
                            log_dir=model_dir, profile_batch=profile_batch,
                        ),
                    ]
                )

            logging.info(
                "Training starting with operative config: \n{}".format(
                    gin.operative_config_str()
                )
            )
            model.summary(print_fn=logging.info)

            kwargs = dict(
                x=train_ds,
                epochs=epochs,
                verbose=verbose,
                steps_per_epoch=train_steps,
                validation_data=val_ds,
                validation_steps=val_steps,
                callbacks=used_callbacks,
                initial_epoch=initial_epoch,
                validation_freq=validation_freq,
            )

            if build_only:
                logging.info("Built successfully")
                return None
            return model.fit(**kwargs)

        if tf.executing_eagerly():
            do_fit()
        else:
            with tf.compat.v1.Session():
                do_fit()

    def run_dataset(
        self,
        burn_iters: int = 10,
        min_iters: int = 10,
        split="train",
        callback: Optional[Callable[[Any, Any], None]] = None,
    ):
        logging.info("Running dataset")
        source = self.source
        dataset = source.get_dataset(split)
        if tf.executing_eagerly():
            # for example, label in tqdm(dataset.take(burn_iters), total=burn_iters):
            it = iter(dataset)
            for _ in tqdm.trange(burn_iters, desc="Burning in..."):
                example, label = next(it)
                if callback is not None:
                    callback(example, label)
            # for example, label in tqdm(dataset.take(min_iters), total=min_iters):
            for _ in tqdm.trange(min_iters, desc="Benchmarking..."):
                example, label = next(it)
                if callback is not None:
                    callback(example, label)
        else:
            example, label = tf.compat.v1.data.make_one_shot_iterator(
                dataset
            ).get_next()
            with tf.compat.v1.Session() as sess:
                for _ in tqdm.tqdm(range(burn_iters), total=burn_iters):
                    out = sess.run((example, label))

                    if callback is not None:
                        callback(*out)
                for _ in tqdm.tqdm(range(min_iters), total=min_iters):
                    out = sess.run((example, label))

                    if callback is not None:
                        callback(*out)
        logging.info("Finished running dataset")

    def run_predictions(
        self,
        num_examples: int = 10,
        split="train",
        callback: Optional[Callable[[Any, Any], None]] = None,
    ):
        model = self.model
        logging.info("Running dataset")
        dataset = self.source.get_dataset(split).take(num_examples)
        if tf.executing_eagerly():
            for example, label in tqdm.tqdm(dataset, total=num_examples):
                predictions = model(example, training=False)
                if callback is not None:
                    callback(predictions, label)
        else:
            example, label = tf.compat.v1.data.make_one_shot_iterator(
                dataset
            ).get_next()
            predictions = model(example, training=False)
            with tf.compat.v1.Session() as sess:
                for _ in tqdm.trange(num_examples):
                    out = sess.run((predictions, label))
                    if callback is not None:
                        callback(*out)
        logging.info("Finished running dataset")


@gin.configurable(module="kb.framework")
def fit(
    trainable: Trainable,
    epochs: int,
    verbose: bool = True,
    callbacks: Sequence[tf.keras.callbacks.Callback] = (),
    chkpt_kwargs: Optional[Mapping[str, Any]] = None,
    validation_freq: int = 1,
    profile_batch: Union[int, str, Tuple[int, int]] = 2,
    build_only: bool = False,
):
    return trainable.fit(
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        chkpt_kwargs=chkpt_kwargs,
        validation_freq=validation_freq,
        profile_batch=profile_batch,
        build_only=build_only,
    )


@gin.configurable(module="kb.framework")
def evaluate(trainable: Trainable, chkpt_kwargs: Optional[Mapping[str, Any]] = None):
    return trainable.evaluate(chkpt_kwargs=chkpt_kwargs)


@gin.configurable(module="kb.framework")
def run_dataset(
    trainable: Trainable,
    burn_iters: int = 10,
    min_iters: int = 10,
    split="train",
    callback: Optional[Callable[[Any, Any], None]] = None,
):
    return trainable.run_dataset(
        burn_iters=burn_iters, min_iters=min_iters, split=split, callback=callback,
    )


@gin.configurable(module="kb.framework")
def run_predictions(
    trainable: Trainable,
    num_examples: int = 10,
    split="train",
    callback: Optional[Callable[[Any, Any], None]] = None,
):
    return trainable.run_predictions(num_examples, split=split, callback=callback)


@gin.configurable(module="kb.framework")
def model_summary(trainable: Trainable):
    return trainable.model.model_summary()


@gin.configurable(module="kb.framework")
def operative_config(trainable: Trainable):
    del trainable
    out = gin.operative_config_str()
    print(out)
    return out


@gin.configurable(module="kb.framework")
def model_config(trainable: Trainable):
    out = trainable.model.model_config()
    print(out)
    return out


@gin.configurable(module="kb.framework")
def check_gradients(trainable: Trainable, burn_iters: int = 50, run_iters: int = 5):
    return trainable.check_gradients(burn_iters=burn_iters, run_iters=run_iters)


@gin.configurable(module="kb.framework")
def check_weight_updates(trainable: Trainable, total_steps: int):
    return trainable.check_weight_updates(total_steps)


@gin.configurable(module="kb.framework")
def benchmark(
    trainable=gin.REQUIRED,
    burn_iters=gin.REQUIRED,
    min_iters=gin.REQUIRED,
    dataset_only=False,
    forward_only=False,
    fixed_inputs=False,
):
    return trainable.benchmark(
        burn_iters,
        min_iters,
        dataset_only=dataset_only,
        forward_only=forward_only,
        fixed_inputs=fixed_inputs,
    )


def graph_wrap(fn: Callable):
    with tf.Graph().as_default():
        return fn()


@gin.configurable(module="kb.framework")
def benchmark_wrapped():
    """
    Function to be used to run benchmarks in a graph context.

    `benchmark` takes gin parameters as inputs which partially build the graph.
    By having this wrapper, those arguments aren't created until after we enter
    the graph context.
    """
    return graph_wrap(benchmark)


@gin.configurable(module="kb.framework")
def fit_wrapped():
    return graph_wrap(fit)


@gin.configurable(module="kb.framework.misc")
def print_preds(predictions, labels):
    del labels
    print(predictions)
