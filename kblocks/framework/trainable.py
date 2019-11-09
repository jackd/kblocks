from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import os
import tensorflow as tf
import gin
from typing import Sequence, Mapping, Any, Optional, Callable, List

from kblocks.framework.problems import Problem
from kblocks.framework.pipelines import Pipeline
from kblocks import callbacks as cb
from kblocks.tf_typing import NestedTensorLikeSpec
from kblocks.framework.problems.core import Split
from kblocks.optimizers.scope import OptimizerScope
# from kblocks.callbacks.log_updater import LogUpdater


@gin.configurable(module='kb.framework')
class Trainable(object):

    def __init__(
            self,
            problem: Problem,
            pipeline_fn: Callable[[NestedTensorLikeSpec, NestedTensorLikeSpec],
                                  Pipeline],
            optimizer_fn: Optional[
                Callable[[], tf.keras.optimizers.Optimizer]] = None,
            model_dir: Optional[str] = None):
        # self._log_updater = LogUpdater()
        if model_dir is not None:
            model_dir = os.path.expanduser(os.path.expandvars(model_dir))
        self._model_dir = model_dir
        self._problem = problem
        # with self._log_updater:
        with problem:
            optimizer = None if optimizer_fn is None else optimizer_fn()
            with OptimizerScope(optimizer):
                self._pipeline = pipeline_fn(problem.features_spec,
                                             problem.outputs_spec)
            if self._pipeline.model is not None:
                self._pipeline.model.compile(
                    loss=self._problem.loss,
                    metrics=self._problem.metrics,
                    optimizer=optimizer,
                )

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
                      num_parallel_calls=tf.data.experimental.AUTOTUNE):
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
            if split == 'train':
                dataset = dataset.shuffle(shuffle_buffer or
                                          problem.shuffle_buffer)
            dataset = dataset.repeat()
            dataset = dataset.map(pre_batch_map, num_parallel_calls)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(post_batch_map, num_parallel_calls)
            return dataset

        return tf.nest.map_structure(prep_dataset, splits)

    def fit(self,
            batch_size: int,
            epochs: Optional[int] = None,
            total_train_steps: Optional[int] = None,
            shuffle_buffer: Optional[int] = None,
            verbose: bool = True,
            callbacks: List[tf.keras.callbacks.Callback] = [],
            chkpt_kwargs: Mapping[str, Any] = {}):
        problem = self.problem
        pipeline = self.pipeline
        model_dir = self.model_dir

        if (epochs is None) == (total_train_steps is None):
            raise ValueError('At least one of epoch or steps must be supplied')

        splits = ('train', 'validation')
        train_ds, val_ds = self._get_datasets(splits, batch_size,
                                              shuffle_buffer)
        train_steps, val_steps = (
            problem.examples_per_epoch(split, batch_size) for split in splits)

        used_callbacks = [
            # self._log_updater,
            cb.AbslLogger(),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        model = pipeline.model
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
                chkpt_callback.restore(initial_epoch)

            used_callbacks.append(chkpt_callback)
        else:
            initial_epoch = 0

        used_callbacks.extend(callbacks)
        if model_dir is not None:
            callbacks.extend([
                cb.HPCallback(log_dir=model_dir),
                tf.keras.callbacks.TensorBoard(log_dir=model_dir,
                                               profile_batch=train_steps // 2),
            ])
        # if verbose:
        #     callbacks.append(tf.keras.callbacks.ProgbarLogger())

        logging.info('Training starting with operative config: \n{}'.format(
            gin.operative_config_str()))
        model.summary(print_fn=logging.info)

        if epochs is None:
            epochs = total_train_steps // train_steps

        return model.fit(
            train_ds,
            epochs=epochs,
            verbose=verbose,
            steps_per_epoch=train_steps,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=used_callbacks,
            initial_epoch=initial_epoch,
        )

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


@gin.configurable(module='kb.framework')
def fit(trainable: Trainable,
        batch_size: int,
        epochs: Optional[int] = None,
        total_train_steps: Optional[int] = None,
        shuffle_buffer: Optional[int] = None,
        verbose: bool = True,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        chkpt_kwargs: Mapping[str, Any] = {}):
    return trainable.fit(batch_size=batch_size,
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
                callback: Optional[Callable[[Any, Any], None]] = None):
    trainable.run_dataset(batch_size,
                          num_examples,
                          shuffle_buffer=shuffle_buffer,
                          split=split,
                          callback=callback)


@gin.configurable(module='kb.framework')
def model_summary(trainable: Trainable):
    trainable.model_summary()
