from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import os
from typing import Sequence, Mapping, Any, Optional, Callable, List
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gin

from kblocks.framework.problems import Problem
from kblocks.framework.problems import scope as problem_scope
from kblocks.framework.pipelines import Pipeline
from kblocks.extras import callbacks as cb
from kblocks.tf_typing import NestedTensorLikeSpec
from kblocks.framework.problems.core import Split
from kblocks.framework import steps
from kblocks.extras.callbacks import log_updater as log_lib
from kblocks.extras.callbacks import value_updater as val_lib
# from kblocks.extras.callbacks.step import StepFnCallback
from kblocks.benchmark_utils import summarize

# class DebugCallback(tf.keras.callbacks.Callback):

#     def on_batch_end(self, batch, logs=None):
#         print('batch end', batch)
#         print(logs)

#     def on_epoch_end(self, epoch, logs=None):
#         print('epoch end', epoch)
#         print(logs)
# raise Exception()


def _get_epochs(epochs: Optional[int], total_train_steps: Optional[int],
                train_steps: int):
    msg = 'Exactly one of epochs or total_train_steps must be supplied'
    if epochs is None:
        if total_train_steps is None:
            raise ValueError(msg)
        epochs = total_train_steps // train_steps
    elif total_train_steps is not None:
        raise ValueError(msg)
    return epochs


@gin.configurable(module='kb.framework')
class Trainable(object):

    def __init__(
            self,
            problem: Problem,
            pipeline_fn: Callable[[NestedTensorLikeSpec, NestedTensorLikeSpec],
                                  Pipeline],
            optimizer_fn: Optional[
                Callable[[], tf.keras.optimizers.Optimizer]] = None,
            model_dir: Optional[str] = None,
            add_learning_rate_summary: bool = False,
            auto_compile=True):
        self._log_updater = log_lib.LogUpdater()
        self._value_updater = val_lib.ValueUpdater()
        # self._step_fn_callback = StepFnCallback()
        if model_dir is not None:
            model_dir = os.path.expanduser(os.path.expandvars(model_dir))
        self._model_dir = model_dir
        self._problem = problem
        self._optimizer_fn = optimizer_fn
        with log_lib.scope(self._log_updater):
            with val_lib.scope(self._value_updater):
                with problem_scope(problem):
                    if optimizer_fn is None:
                        optimizer = None
                    else:
                        optimizer = optimizer_fn()
                        steps.set_step(optimizer.iterations)
                    if add_learning_rate_summary and optimizer is not None:
                        self._log_updater.log_each_epoch(
                            'learning_rate', optimizer._decayed_lr(tf.float32))
                    self._pipeline = pipeline_fn(problem.features_spec,
                                                 problem.outputs_spec)
                self._optimizer = optimizer
                if auto_compile and self._pipeline.model is not None:
                    self._compile()

    def _compile(self):
        self._pipeline.model.compile(loss=self._problem.loss,
                                     metrics=self._problem.metrics,
                                     optimizer=self._optimizer)

    def operative_config(self):
        return gin.operative_config_str()

    def model_config(self):
        return self._pipeline.model.get_config()

    def model_summary(self):
        self._pipeline.model.summary()

    @property
    def model_dir(self) -> Optional[str]:
        return self._model_dir

    @property
    def problem(self) -> Problem:
        return self._problem

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def _get_datasets(self,
                      splits,
                      batch_size,
                      shuffle_buffer=None,
                      num_parallel_calls: Optional[int] = None,
                      prefetch_buffer: Optional[int] = None):
        if num_parallel_calls is None:
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        if prefetch_buffer is None:
            prefetch_buffer = tf.data.experimental.AUTOTUNE
        problem = self.problem
        pipeline = self.pipeline

        def pre_batch_map(features, labels, weights=None):
            features = pipeline.pre_batch_map(features)
            return (features, labels) if weights is None else (features, labels,
                                                               weights)

        def post_batch_map(features, labels, weights=None):
            features = pipeline.post_batch_map(features)
            args = problem.post_batch_map(labels, weights)
            if isinstance(args, Sequence) and len(args) == 2:
                labels, weights = args
                return features, labels, weights
            else:
                return features, args

        def prep_dataset(split):
            dataset = problem.get_base_dataset(split=split)
            if split == 'train' and shuffle_buffer != -1:
                dataset = dataset.shuffle(shuffle_buffer or
                                          problem.shuffle_buffer)
            dataset = dataset.repeat()
            dataset = dataset.map(pre_batch_map, num_parallel_calls)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.map(post_batch_map, num_parallel_calls)
            dataset = dataset.prefetch(prefetch_buffer)
            return dataset

        return tf.nest.map_structure(prep_dataset, splits)

    def check_weight_updates(self, batch_size: int, total_steps: int):
        model = self._pipeline.model
        weight_vals = [w.numpy() for w in model.trainable_weights]
        ds = self._get_datasets('train',
                                batch_size=batch_size,
                                shuffle_buffer=total_steps).take(total_steps)

        model.fit(ds, steps_per_epoch=total_steps, epochs=1)
        for weight, orig in zip(model.trainable_weights, weight_vals):
            if np.allclose(weight.numpy(), orig):
                logging.info('{} ({}) value largely unchanged'.format(
                    weight.name, tuple(weight.shape)))

    def check_gradients(self, batch_size: int, burn_iters=50, run_iters=5):
        model = self._pipeline.model
        loss: tf.keras.losses.Loss = self._problem.loss
        dataset = self._get_datasets('train',
                                     batch_size,
                                     shuffle_buffer=None,
                                     prefetch_buffer=None)
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

        for args in tqdm(dataset.take(burn_iters),
                         total=burn_iters,
                         desc='Burning in...'):
            grads_and_vars = compute_grads_and_vars(args)
            model.optimizer.apply_gradients(grads_and_vars)

        for i, args in enumerate(dataset.take(run_iters)):
            print('Starting run step {}'.format(i))
            grads_and_vars = compute_grads_and_vars(args)
            for grad, weight in grads_and_vars:
                if grad is None:
                    print('Gradient for weight {} ({}) is None'.format(
                        weight.name, tuple(weight.shape)))
                elif np.allclose(grad.numpy(), 0):
                    print(
                        'Gradients for weight {} ({}) are all close to zero, largest is {}'
                        .format(weight.name, tuple(weight.shape),
                                tf.reduce_max(tf.abs(grad)).numpy()))
        print('Finished checking gradients')

    def benchmark(self,
                  batch_size: int,
                  burn_iters: int,
                  min_iters: int,
                  prefetch_buffer: Optional[int] = None,
                  dataset_only: bool = False,
                  forward_only: bool = False,
                  fixed_inputs: bool = False):
        kwargs = dict(
            batch_size=batch_size,
            burn_iters=burn_iters,
            min_iters=min_iters,
            prefetch_buffer=prefetch_buffer,
        )
        if dataset_only:
            return self.benchmark_dataset(**kwargs)
        else:
            return self.benchmark_pipeline(forward_only=forward_only,
                                           fixed_inputs=fixed_inputs,
                                           **kwargs)

    def benchmark_dataset(self,
                          batch_size: int,
                          burn_iters: int,
                          min_iters: int,
                          prefetch_buffer: Optional[int] = None):
        dataset = self._get_datasets('train',
                                     batch_size,
                                     -1,
                                     prefetch_buffer=prefetch_buffer)
        op = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        bm = tf.test.Benchmark()
        with tf.compat.v1.Session() as sess:
            logging.info('Starting benchmarking...')
            result = bm.run_op_benchmark(sess,
                                         op,
                                         burn_iters=burn_iters,
                                         min_iters=min_iters)
            summarize(result)
        return result

    def benchmark_pipeline(self,
                           batch_size: int,
                           burn_iters: int,
                           min_iters: int,
                           prefetch_buffer: Optional[int] = None,
                           forward_only=False,
                           fixed_inputs=False):
        pipeline = self._pipeline
        with tf.Graph().as_default():
            dataset = self._get_datasets(
                'train',
                batch_size,
                -1,
                prefetch_buffer=prefetch_buffer,
            )

            inputs, labels = tf.compat.v1.data.make_one_shot_iterator(
                dataset).get_next()

            if fixed_inputs:
                logging.info('Getting hack inputs...')
                with tf.compat.v1.Session() as sess:
                    inputs, labels = sess.run((inputs, labels))
                logging.info('Got inputs!')

                inputs, labels = tf.nest.map_structure(
                    lambda x: tf.constant(x)
                    if isinstance(x, np.ndarray) else tf.RaggedTensor.
                    from_nested_row_splits(tf.constant(x.flat_values), [
                        tf.constant(i) for i in x.nested_row_splits
                    ]), (inputs, labels))

            model = tf.keras.models.clone_model(pipeline.model,
                                                input_tensors=inputs)

            out, = model.outputs
            if forward_only:
                train_op = out
            else:
                # recreate loss/optimizer for new graph
                serialized_loss = tf.keras.utils.serialize_keras_object(
                    self.problem.loss)
                loss_ = tf.keras.losses.deserialize(serialized_loss)
                optimizer = self._optimizer_fn()
                loss = loss_(labels, out)
                weights = model.trainable_weights
                grads = optimizer.get_gradients(loss, weights)
                grads_and_vars = tuple(
                    (g, v) for g, v in zip(grads, weights) if g is not None)
                train_op = optimizer.apply_gradients(grads_and_vars)
                train_op = tf.group((train_op,) + tuple(model.updates))

            bm = tf.test.Benchmark()
            with tf.compat.v1.Session() as sess:
                logging.info('Initializing variables...')

                sess.run(tf.compat.v1.global_variables_initializer())

                logging.info('Starting benchmarking...')
                result = bm.run_op_benchmark(sess,
                                             train_op,
                                             burn_iters=burn_iters,
                                             min_iters=min_iters)
        summarize(result)
        return result

    def profile(self,
                batch_size: int,
                burn_iters: int,
                min_iters: int,
                prefetch_buffer: Optional[int] = None):
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/profiler/Profiler
        from kblocks.profile_utils import summarize
        Profiler = tf.compat.v1.profiler.Profiler
        ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder
        model = self._pipeline.model
        problem = self.problem
        optimizer: tf.keras.optimizers.Optimizer = model.optimizer
        dataset = self._get_datasets('train',
                                     batch_size,
                                     -1,
                                     prefetch_buffer=prefetch_buffer)
        inputs, labels = tf.compat.v1.data.make_one_shot_iterator(
            dataset).get_next()
        out = model(inputs)
        loss_: tf.keras.losses.Loss = problem.loss
        loss = loss_(labels, out)
        weights = model.trainable_weights
        grads = optimizer.get_gradients(loss, weights)
        grads_and_vars = tuple(
            (g, v) for g, v in zip(grads, weights) if g is not None)
        train_op = optimizer.apply_gradients(grads_and_vars)
        all_ops = (train_op,) + tuple(model.updates)

        with tf.compat.v1.Session() as sess:
            logging.info('Starting profiling...')
            profiler = Profiler(graph=sess.graph)

            variables = model.weights + optimizer.weights
            # TODO: how do you get optimizer hyperparameter variables?
            for name in ('beta_1', 'beta_2', 'learning_rate', 'momentum'):
                a = getattr(optimizer, name, None)
                if isinstance(a, tf.Variable):
                    variables.append(a)
            sess.run([v.initializer for v in variables])
            for i in range(burn_iters):
                sess.run(all_ops)

            run_meta = tf.compat.v1.RunMetadata()

            opts = ProfileOptionBuilder.time_and_memory(
                # min_accelerator_micros=1
            )
            profiles = []
            for i in range(min_iters):
                sess.run(train_op,
                         options=tf.compat.v1.RunOptions(
                             trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                         run_metadata=run_meta)
                profiler.add_step(i, run_meta)

                # Profile the parameters of your model.
                # profiler.profile_name_scope(options=(
                #     ProfileOptionBuilder.trainable_variables_parameter()))

                # Or profile the timing of your model operations.
                profiles.append(profiler.profile_operations(options=opts))

                # # Or you can generate a timeline:
                # opts = (option_builder.ProfileOptionBuilder(
                #         option_builder.ProfileOptionBuilder.time_and_memory())
                #         .with_step(i)
                #         .with_timeline_output(filename).build())
                # profiler.profile_graph(options=opts)
            ALL_ADVICE = {
                'ExpensiveOperationChecker': {},
                'AcceleratorUtilizationChecker': {},
                'JobChecker': {},  # Only available internally.
                'OperationChecker': {},
            }
            profiler.advise(ALL_ADVICE)
            summarize(profiles)
            return profiles

    def custom_fit(self,
                   batch_size: int,
                   epochs: Optional[int] = None,
                   total_train_steps: Optional[int] = None,
                   shuffle_buffer: Optional[int] = None,
                   verbose: bool = True,
                   callbacks: List[tf.keras.callbacks.Callback] = [],
                   chkpt_kwargs: Mapping[str, Any] = {}):
        from tqdm import tqdm
        if len(callbacks) > 0:
            raise NotImplementedError('TODO: add callback support')
        problem = self.problem
        pipeline = self.pipeline
        # model_dir = self.model_dir
        model = pipeline.model
        trainable_weights = model.trainable_weights
        metrics: List[tf.keras.metrics.Metric] = problem.metrics
        loss: tf.keras.losses.Loss = problem.loss
        optimizer: tf.keras.optimizers.Optimizer = model.optimizer

        splits = ('train', 'validation')
        train_ds, val_ds = self._get_datasets(splits, batch_size,
                                              shuffle_buffer)
        train_steps, val_steps = (
            problem.examples_per_epoch(split, batch_size) for split in splits)

        epochs = _get_epochs(epochs, total_train_steps, train_steps)
        for epoch in range(epochs):
            # train loop
            tf.keras.backend.set_learning_phase(True)
            for metric in metrics:
                metric.reset_states()

            for example, labels in tqdm(train_ds.take(train_steps),
                                        desc='Training epoch {} / {}'.format(
                                            epoch, epochs),
                                        total=train_steps):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(trainable_weights)
                    preds = model(example)
                    loss_val = loss(labels, preds)
                    if model.losses:
                        loss_val = tf.add_n(model.losses) + loss_val
                    grads = tape.gradient(loss_val,
                                          trainable_weights,
                                          unconnected_gradients='zero')
                optimizer.apply_gradients(zip(grads, trainable_weights))
                for m in metrics:
                    m.update_state(labels, preds)
            logging.info('Finished training epoch {} / {}')
            for m in metrics:
                logging.info('{}: {}'.format(m.name, m.result()))

            # val loop
            tf.keras.backend.set_learning_phase(False)
            for m in metrics:
                m.reset_states()

            for example, labels in tqdm(val_ds.take(val_steps),
                                        desc='Evaluating epoch {} / {}'.format(
                                            epoch, epochs),
                                        total=val_steps):
                preds = model(example)
                for m in metrics:
                    m.update_state(labels, preds)
            logging.info('Finished evaluating epoch {} / {}')
            for m in metrics:
                logging.info('{}: {}'.format(m.name, m.result()))

    def evaluate(self, batch_size: int, chkpt_kwargs: Mapping[str, Any] = {}):
        problem = self.problem
        pipeline = self.pipeline
        model_dir = self.model_dir
        model = pipeline.model
        split = 'validation'

        ds = self._get_datasets(split, batch_size)
        steps = problem.examples_per_epoch(split) // batch_size

        if model_dir is None:
            logging.warning(
                'No model_dir provided - evaluating without restoration')
        else:
            chkpt_dir = os.path.join(model_dir, 'chkpts')
            chkpt_callback = cb.CheckpointCallback(directory=chkpt_dir,
                                                   **chkpt_kwargs)

            chkpt_callback.set_model(model)
            chkpt = chkpt_callback.checkpoint()
            if chkpt is None:
                logging.warning(
                    'No checkpoint found - evaluating without restoration')
            else:
                initial_epoch = chkpt_callback.epoch(chkpt)
                chkpt_callback.restore(initial_epoch).expect_partial()
        model.evaluate(ds, steps=steps)

    def fit_simple(self,
                   batch_size: int,
                   epochs: Optional[int] = None,
                   total_train_steps: Optional[int] = None,
                   shuffle_buffer: Optional[int] = None,
                   verbose: bool = True,
                   callbacks: List[tf.keras.callbacks.Callback] = [],
                   chkpt_kwargs: Mapping[str, Any] = {},
                   validation_freq: int = 1):
        pipeline = self._pipeline
        train_steps = self.problem.examples_per_epoch('train', batch_size)
        serialized_loss = tf.keras.utils.serialize_keras_object(
            self.problem.loss)
        model = pipeline.model
        with tf.Graph().as_default():
            # recreate model for new graph
            dataset = self._get_datasets('train', batch_size)
            inputs, labels = tf.compat.v1.data.make_one_shot_iterator(
                dataset).get_next()
            # with tf.compat.v1.Session():
            model = tf.keras.models.clone_model(model, input_tensors=inputs)
            model.compile(optimizer=self._optimizer_fn(),
                          loss=tf.keras.losses.deserialize(serialized_loss),
                          target_tensors=labels)
            model.fit(
                # dataset,
                epochs=epochs,
                steps_per_epoch=train_steps,
            )

        # problem = self.problem
        # pipeline = self.pipeline
        # model = pipeline.model
        # # model_dir = self.model_dir

        # splits = ('train', 'validation')
        # train_ds, val_ds = self._get_datasets(
        #     splits,
        #     batch_size,
        #     shuffle_buffer,
        # )
        # train_steps, val_steps = (
        #     problem.examples_per_epoch(split, batch_size) for split in splits)

        # epochs = _get_epochs(epochs, total_train_steps, train_steps)
        # model.fit(
        #     x=train_ds,
        #     epochs=epochs,
        #     verbose=verbose,
        #     steps_per_epoch=train_steps,
        #     validation_data=val_ds,
        #     validation_steps=val_steps,
        #     # callbacks=used_callbacks,
        #     # initial_epoch=initial_epoch,
        #     validation_freq=validation_freq,
        # )

    def fit(self,
            batch_size: int,
            epochs: Optional[int] = None,
            total_train_steps: Optional[int] = None,
            shuffle_buffer: Optional[int] = None,
            verbose: bool = True,
            callbacks: List[tf.keras.callbacks.Callback] = [],
            chkpt_kwargs: Mapping[str, Any] = {},
            validation_freq: int = 1,
            use_custom: bool = False):
        problem = self.problem
        pipeline = self.pipeline
        model_dir = self.model_dir

        splits = ('train', 'validation')
        train_ds, val_ds = self._get_datasets(
            splits,
            batch_size,
            shuffle_buffer,
        )
        train_steps, val_steps = (
            problem.examples_per_epoch(split, batch_size) for split in splits)

        epochs = _get_epochs(epochs, total_train_steps, train_steps)

        used_callbacks: List[tf.keras.callbacks.Callback] = [
            self._value_updater
        ] if self._value_updater.used else []
        used_callbacks.extend([
            # DebugCallback(),
            self._log_updater,
            # self._step_fn_callback,
            cb.AbslLogger(),
            tf.keras.callbacks.TerminateOnNaN(),
        ])

        model = pipeline.model

        # creates relevant learning_rate variable for restoration
        model.optimizer.learning_rate

        # do_fit needs to be in a session if running in graph mode
        def do_fit():
            if model_dir is not None:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)
                chkpt_dir = os.path.join(model_dir, 'chkpts')
                chkpt_callback = cb.CheckpointCallback(directory=chkpt_dir,
                                                       **chkpt_kwargs)

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
                    tf.compat.v1.global_variables_initializer())

            used_callbacks.extend(callbacks)
            if model_dir is not None:
                used_callbacks.extend([
                    # cb.HPCallback(log_dir=model_dir),
                    tf.keras.callbacks.TensorBoard(
                        log_dir=model_dir,
                        profile_batch=train_steps // 2,
                        # profile_batch=0,
                    ),
                ])

            logging.info('Training starting with operative config: \n{}'.format(
                gin.operative_config_str()))
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

            if use_custom:
                from kblocks.model_utils import custom_fit
                return custom_fit(model, **kwargs)
            else:
                return model.fit(**kwargs)

        # checkpoints require an active session
        if tf.executing_eagerly():
            do_fit()
        else:
            with tf.compat.v1.Session():
                do_fit()

    def run_dataset(self,
                    batch_size: int,
                    num_examples: int = 10,
                    shuffle_buffer: Optional[int] = None,
                    split: Split = 'train',
                    num_parallel_calls: int = 1,
                    callback: Optional[Callable[[Any, Any], None]] = None):
        from tqdm import tqdm
        logging.info('Running dataset')
        dataset = self._get_datasets(
            split,
            batch_size,
            shuffle_buffer,
            num_parallel_calls=num_parallel_calls).take(num_examples)
        if tf.executing_eagerly():
            for example, label in tqdm(dataset, total=num_examples):
                if callback is not None:
                    callback(example, label)
        else:
            example, label = tf.compat.v1.data.make_one_shot_iterator(
                dataset).get_next()
            with tf.compat.v1.Session() as sess:
                for _ in tqdm(range(num_examples), total=num_examples):
                    out = sess.run((example, label))

                    if callback is not None:
                        callback(*out)
        logging.info('Finished running dataset')

    def run_predictions(self,
                        batch_size: int,
                        num_examples: int = 10,
                        shuffle_buffer: Optional[int] = None,
                        split: Split = 'train',
                        num_parallel_calls: int = 1,
                        callback: Optional[Callable[[Any, Any], None]] = None):
        from tqdm import tqdm
        model = self._pipeline.model
        logging.info('Running dataset')
        dataset = self._get_datasets(
            split,
            batch_size,
            shuffle_buffer,
            num_parallel_calls=num_parallel_calls).take(num_examples)
        if tf.executing_eagerly():
            for example, label in tqdm(dataset, total=num_examples):
                predictions = model(example)
                if callback is not None:
                    callback(predictions, label)
        else:
            example, label = tf.compat.v1.data.make_one_shot_iterator(
                dataset).get_next()
            predictions = model(example)
            with tf.compat.v1.Session() as sess:
                for _ in tqdm(range(num_examples), total=num_examples):
                    out = sess.run((predictions, label))
                    if callback is not None:
                        callback(*out)
        logging.info('Finished running dataset')


@gin.configurable(module='kb.framework')
def fit(trainable: Trainable,
        batch_size: int,
        epochs: Optional[int] = None,
        total_train_steps: Optional[int] = None,
        shuffle_buffer: Optional[int] = None,
        verbose: bool = True,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        chkpt_kwargs: Mapping[str, Any] = {},
        validation_freq: int = 1,
        use_custom: bool = False):
    return trainable.fit(batch_size=batch_size,
                         epochs=epochs,
                         total_train_steps=total_train_steps,
                         shuffle_buffer=shuffle_buffer,
                         verbose=verbose,
                         callbacks=callbacks,
                         chkpt_kwargs=chkpt_kwargs,
                         validation_freq=validation_freq,
                         use_custom=use_custom)


@gin.configurable(module='kb.framework')
def fit_simple(trainable: Trainable,
               batch_size: int,
               epochs: Optional[int] = None,
               total_train_steps: Optional[int] = None,
               shuffle_buffer: Optional[int] = None,
               verbose: bool = True,
               callbacks: List[tf.keras.callbacks.Callback] = [],
               chkpt_kwargs: Mapping[str, Any] = {},
               validation_freq: int = 1):
    return trainable.fit_simple(batch_size=batch_size,
                                epochs=epochs,
                                total_train_steps=total_train_steps,
                                shuffle_buffer=shuffle_buffer,
                                verbose=verbose,
                                callbacks=callbacks,
                                chkpt_kwargs=chkpt_kwargs,
                                validation_freq=validation_freq)


@gin.configurable(module='kb.framework')
def evaluate(trainable: Trainable,
             batch_size: int,
             chkpt_kwargs: Mapping[str, Any] = {}):
    return trainable.evaluate(batch_size=batch_size, chkpt_kwargs=chkpt_kwargs)


@gin.configurable(module='kb.framework')
def custom_fit(trainable: Trainable,
               batch_size: int,
               epochs: Optional[int] = None,
               total_train_steps: Optional[int] = None,
               shuffle_buffer: Optional[int] = None,
               verbose: bool = True,
               callbacks: List[tf.keras.callbacks.Callback] = [],
               chkpt_kwargs: Mapping[str, Any] = {}):
    return trainable.custom_fit(batch_size=batch_size,
                                epochs=epochs,
                                total_train_steps=total_train_steps,
                                shuffle_buffer=shuffle_buffer,
                                verbose=verbose,
                                callbacks=callbacks,
                                chkpt_kwargs=chkpt_kwargs)


@gin.configurable(module='kb.framework')
def run_dataset(trainable: Trainable,
                batch_size: int,
                num_examples: int = 10,
                shuffle_buffer: Optional[int] = None,
                split: Split = 'train',
                callback: Optional[Callable[[Any, Any], None]] = None,
                num_parallel_calls=1):
    return trainable.run_dataset(batch_size,
                                 num_examples,
                                 shuffle_buffer=shuffle_buffer,
                                 split=split,
                                 callback=callback,
                                 num_parallel_calls=num_parallel_calls)


@gin.configurable(module='kb.framework')
def run_predictions(trainable: Trainable,
                    batch_size: int,
                    num_examples: int = 10,
                    shuffle_buffer: Optional[int] = None,
                    split: Split = 'train',
                    callback: Optional[Callable[[Any, Any], None]] = None):
    return trainable.run_predictions(batch_size,
                                     num_examples,
                                     shuffle_buffer=shuffle_buffer,
                                     split=split,
                                     callback=callback)


@gin.configurable(module='kb.framework')
def model_summary(trainable: Trainable):
    return trainable.model_summary()


@gin.configurable(module='kb.framework')
def operative_config(trainable: Trainable):
    out = trainable.operative_config()
    print(out)
    return out


@gin.configurable(module='kb.framework')
def model_config(trainable: Trainable):
    out = trainable.model_config()
    print(out)
    return out


@gin.configurable(module='kb.framework')
def check_gradients(trainable: Trainable,
                    batch_size: int,
                    burn_iters: int = 50,
                    run_iters: int = 5):
    return trainable.check_gradients(batch_size,
                                     burn_iters=burn_iters,
                                     run_iters=run_iters)


@gin.configurable(module='kb.framework')
def check_weight_updates(trainable: Trainable, batch_size: int,
                         total_steps: int):
    return trainable.check_weight_updates(batch_size, total_steps)


@gin.configurable(module='kb.framework')
def benchmark(trainable=gin.REQUIRED,
              batch_size=gin.REQUIRED,
              burn_iters=gin.REQUIRED,
              min_iters=gin.REQUIRED,
              prefetch_buffer=None,
              dataset_only=False,
              forward_only=False,
              fixed_inputs=False):
    return trainable.benchmark(batch_size,
                               burn_iters,
                               min_iters,
                               prefetch_buffer,
                               dataset_only=dataset_only,
                               forward_only=forward_only,
                               fixed_inputs=fixed_inputs)


def graph_wrap(fn: Callable):
    with tf.Graph().as_default():
        return fn()


@gin.configurable(module='kb.framework')
def benchmark_wrapped():
    """
    Function to be used to run benchmarks in a graph context.

    `benchmark` takes gin parameters as inputs which partially build the graph.
    By having this wrapper, those arguments aren't created until after we enter
    the graph context.
    """
    return graph_wrap(benchmark)


@gin.configurable(module='kb.framework')
def fit_wrapped():
    return graph_wrap(fit)


@gin.configurable(module='kb.framework')
def fit_simple_wrapped():
    return graph_wrap(fit_simple)


@gin.configurable(module='kb.framework')
def profile(trainable=gin.REQUIRED,
            batch_size=gin.REQUIRED,
            burn_iters=gin.REQUIRED,
            min_iters=gin.REQUIRED,
            prefetch_buffer=None):
    return trainable.profile(batch_size=batch_size,
                             burn_iters=burn_iters,
                             min_iters=min_iters,
                             prefetch_buffer=prefetch_buffer)


@gin.configurable(module='kb.framework')
def profile_wrapped():
    return graph_wrap(profile)


@gin.configurable(module='kb.framework.misc')
def print_preds(predictions, labels):
    print(predictions)
