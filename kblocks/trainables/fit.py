from typing import Iterable, Optional

import gin
import tensorflow as tf

from kblocks.experiments import Experiment, Status
from kblocks.extras import callbacks as cb
from kblocks.keras import model as model_lib
from kblocks.serialize import register_serializable
from kblocks.trainables.core import Trainable, get


@gin.configurable(module="kb.trainable")
@register_serializable
class Fit(Experiment):
    def __init__(
        self,
        trainable: Trainable,
        epochs: int,
        save_dir: str,
        checkpoint_interval: int = 1,
        validation_freq: int = 1,
        name: Optional[str] = None,
    ):
        self._trainable = get(trainable)
        self._epochs = epochs
        self._checkpoint_interval = checkpoint_interval
        self._validation_freq = validation_freq
        super().__init__(save_dir=save_dir, name=name)

    def get_config(self):
        return dict(
            trainable=tf.keras.utils.serialize_keras_object(self.trainable),
            epochs=self.epochs,
            checkpoint_interval=self._checkpoint_interval,
            validation_freq=self._validation_freq,
        )

    @property
    def trainable(self) -> Trainable:
        return self._trainable

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def tb_dir(self) -> str:
        return self._subdir("logs")

    @property
    def backup_dir(self) -> str:
        return self._subdir("backup")

    def _run(self, start_status: str = Status.NOT_STARTED):
        trainable = self.trainable
        callbacks = (
            *trainable.callbacks,
            # Additional stateless callbacks
            # tf.keras.callbacks.CSVLogger(self.csv_path, append=True),
            # cb.YamlLogger(self.yaml_path, append=True),
            tf.keras.callbacks.TerminateOnNaN(),
            cb.LearningRateLogger(),
            tf.keras.callbacks.TensorBoard(self.tb_dir, profile_batch=(2, 12)),
            cb.AbslLogger(),
            cb.BackupAndRestore(
                self.backup_dir, checkpoint_interval=self._checkpoint_interval
            ),
        )
        fit(
            trainable=self.trainable,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_freq=self._validation_freq,
        )


@gin.register(module="kb.trainable")
def fit(
    trainable: Trainable,
    epochs: int,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    validation_freq: int = 1,
    verbose: bool = True,
) -> tf.keras.callbacks.History:
    val_source = trainable.validation_source
    return model_lib.fit(
        model=trainable.model,
        train_data=trainable.train_source.dataset,
        epochs=epochs,
        validation_data=None if val_source is None else val_source.dataset,
        callbacks=trainable.callbacks + tuple(callbacks),
        validation_freq=validation_freq,
        steps_per_epoch=trainable.steps_per_epoch,
        verbose=verbose,
    )
