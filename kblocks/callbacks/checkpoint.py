"""
Callback wrapper around `tf.train.CheckpointManager` / `tf.train.Checkpoint`.

See also: https://www.tensorflow.org/guide/checkpoint
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import gin

import tensorflow as tf
from typing import Optional, Union


@gin.configurable(module='kb.callbacks')
class CheckpointCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 directory: str,
                 save_freq: int = 1,
                 restore_on_begin: bool = True,
                 max_to_keep: int = 5,
                 keep_checkpoint_every_n_hours: Optional[int] = None,
                 checkpoint_name: str = 'ckpt'):
        self._manager_kwargs = dict(
            directory=directory,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            checkpoint_name=checkpoint_name)
        self._save_freq = save_freq
        self._restore_on_begin = restore_on_begin
        self._checkpoint = None
        self._manager = None
        self._restored = False

    def set_model(self, model: tf.keras.Model):
        self._restored = False
        self._checkpoint = tf.train.Checkpoint(model=model)
        self._manager = tf.train.CheckpointManager(self._checkpoint,
                                                   **self._manager_kwargs)
        super(CheckpointCallback, self).set_model(model)

    def checkpoint(self, epoch: Optional[int] = None) -> Optional[str]:
        if epoch is None:
            return self._manager.latest_checkpoint
        else:
            chkpt = '{}-{:d}'.format(self._manager._checkpoint_prefix, epoch)
            chkpts = self._manager.checkpoints
            if chkpt not in chkpts:
                raise ValueError(
                    'chkpt for epoch {} not in saved chkpts {}'.format(
                        epoch, chkpts))
            return chkpt

    def epoch(self, chkpt: str) -> Optional[int]:
        return int(chkpt.split('-')[-1])

    def restore(self, epoch_or_chkpt: Optional[Union[int, str]] = None):
        self._restored = True
        if isinstance(epoch_or_chkpt, int):
            chkpt = self.checkpoint(epoch_or_chkpt)
        elif epoch_or_chkpt is None:
            chkpt = self.checkpoint(epoch_or_chkpt)
        else:
            chkpt = epoch_or_chkpt
        if chkpt is None:
            logging.info('No previous checkpoints found. Skipping restoration')
            return None
        epoch = self.epoch(chkpt)
        logging.info('Restoring model at epoch {} from {}'.format(epoch, chkpt))
        return self._checkpoint.restore(chkpt)

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self._save_freq == 0:
            logging.info('Saving model at epoch {}'.format(epoch))
            self._manager.save(epoch)

    def on_train_begin(self, logs=None):
        if self._restore_on_begin:
            self.restore()
        return logs

    def on_test_begin(self, logs=None):
        if not self._restored and self._restore_on_begin:
            self.restore()
        return logs

    def on_predict_begin(self, logs=None):
        if not self._restored and self._restore_on_begin:
            self.restore()
        return logs
