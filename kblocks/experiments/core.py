import abc
import json
import os
from typing import Union

import gin
import psutil
import tensorflow as tf
from absl import logging

from kblocks.experiments import callbacks as cb, Status
from kblocks.experiments import status as status_lib
from kblocks.path import expand


def _status_path(save_dir: str):
    return os.path.join(save_dir, "status.json")


def _operative_config_path(save_dir: str):
    return os.path.join(save_dir, "operative-config.gin")


@gin.configurable(module="kb.experiments")
class Experiment(abc.ABC):
    """
    Class for tracking `gin.operative_config_str`s and experiment statuses.

    Implementations must implement `_run`, which should not call any configurable
    functions. Constructor arguments may be configurable. This ensures the logged
    config string is complete.

    Example usage:
    ```python
    def fit(model: tf.keras.Model, train_data: tf.data.Dataset, epochs: int):
        model.fit(train_data, epochs=epochs)


    @gin.configurable
    class Fit(Experiment):
        def __init__(
                self,
                model: tf.keras.Model,
                train_data: tf.data.Dataset,
                epochs: int,
                experiment_dir: str):
            self.model = model
            self.train_data = train_data
            self.epochs = epochs
            super().__init__(experiment_dir=experiment_dir)

        def _run(self, start_status):
            fit(model=self.model, train_data=self.train_data, epochs=self.epochs)


    config_str = '''\
    run.experiment = @Fit()
    Fit.model = @my_model_fn()
    my_model_fn.num_classes = 10  # nested configuration is fine
    Fit.train_data = @my_train_data_fn()
    Fit.epochs = 5
    Fit.experiment_dir = '/tmp/'
    '''

    gin.parse_config(config_str)
    gin.finalize()
    run()
    ```
    """

    def __init__(self, experiment_dir: str):
        experiment_dir = expand(experiment_dir)
        self._experiment_dir = experiment_dir
        tf.io.gfile.makedirs(self._experiment_dir)
        logging.info(f"Running experiment in {experiment_dir}")

    @property
    def experiment_dir(self) -> str:
        return self._experiment_dir

    @property
    def status_path(self) -> str:
        return _status_path(self._experiment_dir)

    @property
    def operative_config_path(self) -> str:
        return _operative_config_path(self._experiment_dir)

    @property
    def status(self) -> Union[str, Status]:
        path = self.status_path
        if not os.path.isfile(path):
            return status_lib.NotStarted()
        with open(self.status_path, "r") as fp:
            status = status_lib.get(json.load(fp))
        return status

    def _set_status(
        self,
        value: status_lib.Status,
    ):
        assert isinstance(value, status_lib.Status)
        path = self.status_path
        if isinstance(value, status_lib.NotStarted):
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path, "w") as fp:
            json.dump(
                tf.keras.utils.serialize_keras_object(value),
                fp,
                indent=4,
                sort_keys=True,
            )

    def run(self):
        tf.io.gfile.makedirs(self.experiment_dir)
        status = self.status
        if isinstance(status, status_lib.Finished):
            logging.info("Experiment already finished.")
            return

        if isinstance(status, status_lib.Running):
            if psutil.pid_exists(status.pid):
                raise RuntimeError(f"Experiment already underway on pid {status.pid}")

        callbacks = [
            cb.LoggingCallback(),
            cb.OperativeConfigLogger(self.operative_config_path),
        ]

        self._set_status(status_lib.Running(os.getpid()))
        for callback in callbacks:
            callback.on_start(status)

        try:
            self._run(status)
            self._set_status(status_lib.Finished())
            for callback in callbacks:
                callback.on_finished()
        except KeyboardInterrupt:
            self._set_status(status_lib.Interrupted())
            for callback in callbacks:
                callback.on_interrupt()
            raise
        except Exception as exception:
            self._set_status(status_lib.Excepted(exception))
            for callback in callbacks:
                callback.on_exception(exception)
            raise

    @abc.abstractmethod
    def _run(self, start_status: status_lib.Status):
        """Run the experiment from the given starting status."""
        raise NotImplementedError("Abstract method")


@gin.register(module="kb.experiments")
def run(experiment: Experiment):
    return experiment.run()
