# from typing import Optional, Mapping

# import gin
# import tensorflow as tf


# class TrainingState:
#     """
#     Alternative implementation of internal tensorflow `WorkerTrainingState`.

#     Allows for arbitrary modules to be saved, rather than just `model`.
#     """

#     def __init__(
#         self,
#         modules: Mapping[str, tf.Module],
#         checkpoint_dir: str,
#         max_to_keep: int = 1,
#         keep_checkpoint_every_n_hours=None,
#     ):
#         checkpoint = tf.train.Checkpoint(**modules)
#         self._manager = tf.train.CheckpointManager(
#             checkpoint,
#             directory=checkpoint_dir,
#             max_to_keep=max_to_keep,
#             keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
#         )
#         self._saved_epoch = -1

#     @property
#     def saved_epoch(self):
#         return self._saved_epoch

#     def back_up(self, epoch: int):
#         self._manager.save(checkpoint_number=epoch)
#         self._saved_epoch = epoch

#     def restore(self):
#         path = self._manager.restore_or_initialize()
#         if path is not None:
#             self._saved_epoch = int(path.split("-")[-1])

#     def delete_backup(self):
#         tf.io.gfile.rmtree(self._manager.directory)

#     def maybe_load_initial_epoch_from_ckpt(
#           self, initial_epoch: int, mode: str) -> int:
#         if mode == "train" and self._saved_epoch > 0:
#             return self._saved_epoch + 1
#         return initial_epoch


# @gin.configurable(module="kb.callbacks")
# class BackupAndRestore(tf.keras.callbacks.Callback):
#     def __init__(
#         self,
#         directory: str,
#         max_to_keep: int = 1,
#         keep_checkpoint_every_n_hours=None,
#         checkpoint_interval: int = 1,
#         remove_after_training: bool = False,
#         modules: Optional[Mapping[str, tf.Module]] = None,
#     ):
#         if modules:
#             assert "model" not in modules
#         self._modules = modules or {}
#         self._state_kwargs = dict(
#             checkpoint_dir=directory,
#             max_to_keep=max_to_keep,
#             keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
#         )
#         self._checkpoint_interval = checkpoint_interval
#         self._training_state = None
#         self._remove_after_training = remove_after_training
#         self._last_epoch = -1
#         super().__init__()

#     def _save(self, epoch: int):
#         assert epoch != self._training_state.saved_epoch
#         self._training_state.back_up(epoch)

#     def _maybe_save(self, epoch: int, interval=None):
#         last_saved = self._training_state.saved_epoch
#         if epoch - last_saved >= (interval or self._checkpoint_interval):
#             self._training_state.back_up(epoch)

#     def on_train_begin(self, logs=None):
#         state = TrainingState(
#             dict(model=self.model, **self._modules), **self._state_kwargs
#         )
#         self._training_state = state
#         self.model._training_state = state  # pylint: disable=protected-access
#         state.restore()

#     def on_epoch_end(self, epoch: int, logs=None):
#         self._maybe_save(epoch)
#         self._last_epoch = epoch

#     def on_train_end(self, logs=None):
#         if self._remove_after_training:
#             self._training_state.delete_backup()
#         else:
#             self._maybe_save(self._last_epoch, 1)


# def restore(model: tf.keras.Model, backup_dir: str) -> int:
#     chkpt = tf.train.Checkpoint(model=model)
#     manager = tf.train.CheckpointManager(chkpt, backup_dir, 1)
#     path = manager.restore_or_initialize()
#     if path is None:
#         return None
#     return int(path.split("-")[-1])


import gin
import tensorflow as tf

from kblocks.serialize import register_serializable


@gin.configurable(module="kb.callbacks")
@register_serializable
class BackupAndRestore(tf.keras.callbacks.experimental.BackupAndRestore):
    """Generalized version of `tf.keras.callbacks.experimental.BackupAndRestore`."""

    def __init__(
        self,
        backup_dir: str,
        max_to_keep: int = 1,
        remove_after_training: bool = True,
        checkpoint_interval: int = 1,
        keep_checkpoint_every_n_hours=None,
    ):
        super().__init__(backup_dir=backup_dir)
        self._max_to_keep = max_to_keep
        self._remove_after_training = remove_after_training
        self._checkpoint_interval = checkpoint_interval
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self._last_epoch = -1

    def get_config(self):
        return dict(
            backup_dir=self.backup_dir,
            max_to_keep=self._max_to_keep,
            remove_after_training=self._remove_after_training,
            checkpoint_interval=self._checkpoint_interval,
            keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours,
            last_epoch=self._last_epoch,
        )

    def _maybe_save(self, max_interval: int):
        interval = self._last_epoch - self._training_state._ckpt_saved_epoch.numpy()
        if interval >= max_interval:
            self._training_state.back_up(self._last_epoch)

    def on_epoch_end(self, epoch: int, logs=None):
        self._last_epoch = epoch
        self._maybe_save(self._checkpoint_interval)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        manager = self._training_state.write_checkpoint_manager
        manager._max_to_keep = self._max_to_keep
        manager._keep_checkpoint_every_n_hours = self._keep_checkpoint_every_n_hours

    def on_train_end(self, logs=None):
        # pylint: disable=protected-access
        if self._remove_after_training:
            # On exit of training, delete the training state backup file that was saved
            # for the purpose of worker recovery.
            self._training_state.delete_backup()
        else:
            self._maybe_save(1)
        # Clean up the training state.
        del self._training_state
        del self.model._training_state
