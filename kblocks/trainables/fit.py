import os
from typing import Dict, Iterable, Optional

import gin
import numpy as np
import tensorflow as tf
from pandas import read_csv

from kblocks.experiments import Experiment, Status
from kblocks.extras import callbacks as cb
from kblocks.keras import model as model_lib
from kblocks.serialize import register_serializable
from kblocks.trainables.core import Trainable, get


@gin.configurable(module="kb.trainable")
@register_serializable
class Fit(Experiment[Dict[str, np.ndarray]]):
    def __init__(
        self,
        trainable: Trainable,
        epochs: int,
        save_dir: str,
        name: Optional[str] = None,
    ):
        self._trainable = get(trainable)
        self._epochs = epochs
        super().__init__(save_dir=save_dir, name=name)

    def get_config(self):
        return dict(
            trainable=tf.keras.utils.serialize_keras_object(self.trainable),
            epochs=self.epochs,
        )

    @property
    def trainable(self) -> Trainable:
        return self._trainable

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def csv_path(self) -> str:
        return os.path.join(self.results_dir, "results.csv")

    @property
    def tb_dir(self) -> str:
        return self._subdir("logs")

    @property
    def backup_dir(self) -> str:
        return self._subdir("backup")

    def _run(self, start_status: str = Status.NOT_STARTED) -> Dict[str, np.ndarray]:
        trainable = self.trainable
        callbacks = (
            *trainable.callbacks,
            # Additional stateless callbacks
            tf.keras.callbacks.TensorBoard(self.tb_dir, profile_batch=(2, 12)),
            tf.keras.callbacks.CSVLogger(self.csv_path, append=True),
            tf.keras.callbacks.TerminateOnNaN(),
            cb.AbslLogger(),
            tf.keras.callbacks.experimental.BackupAndRestore(self.backup_dir),
        )
        history = fit(
            trainable=self.trainable, epochs=self.epochs, callbacks=callbacks,
        )
        return {k: np.array(v) for k, v in history.history.items()}

    def _save_results(self, results: Dict[str, np.ndarray], results_dir: str):
        del results
        # saved in CSV callback
        path = self.csv_path
        assert os.path.exists(path)
        assert path.startswith(results_dir)

    def _load_results(self, results_dir: str):
        path = self.csv_path
        assert path.startswith(results_dir)
        data = read_csv(path, header=0)
        return data.to_dict()


@gin.register(module="kb.trainable")
def fit(
    trainable: Trainable,
    epochs: int,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    verbose: bool = True,
) -> tf.keras.callbacks.History:
    callbacks = list(callbacks)
    callbacks.extend(trainable.callbacks)
    val_source = trainable.validation_source
    return model_lib.fit(
        model=trainable.model,
        train_data=trainable.train_source.dataset,
        epochs=epochs,
        validation_data=None if val_source is None else val_source.dataset,
        callbacks=callbacks,
        verbose=verbose,
    )
