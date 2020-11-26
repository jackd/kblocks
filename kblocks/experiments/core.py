import abc
import json
import os
from typing import Any, Dict, Generic, Iterable, Optional, TypeVar

import gin
import tensorflow as tf
from absl import logging

from kblocks.experiments import callbacks as cb
from kblocks.experiments.status import Status
from kblocks.path import expand

T = TypeVar("T")


def _serialized_path(save_dir: str):
    return os.path.join(save_dir, "serialized.json")


class Experiment(tf.Module, abc.ABC, Generic[T]):
    def __init__(self, save_dir: str, name: Optional[str] = None):
        self._save_dir = expand(save_dir)
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        tf.Module.__init__(self, name=name)

    @classmethod
    def from_path(cls, path: str):
        if not path.endswith("serialized.json"):
            path = _serialized_path(path)
        path = expand(path)
        if not os.path.isfile(path):
            raise IOError(f"No config file found at {path}")
        with open(path, "r") as fp:
            serialized = json.load(fp)
        experiment = tf.keras.utils.deserialize_keras_object(serialized)
        assert isinstance(experiment, cls)
        return experiment

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def save_dir(self) -> str:
        return self._save_dir

    def _subdir(self, *args) -> str:
        return os.path.join(self.save_dir, *args)

    @property
    def serialized_path(self) -> str:
        return _serialized_path(self.save_dir)

    @property
    def status_path(self) -> str:
        return self._subdir("status.txt")

    @property
    def operative_config_path(self) -> str:
        return self._subdir("operative-config.gin")

    @property
    def initial_checkpoint_prefix(self) -> str:
        return self._subdir("checkpoints", "initial", "chkpt")

    @property
    def final_checkpoint_prefix(self) -> str:
        return self._subdir("checkpoints", "final", "chkpt")

    @property
    def status(self) -> str:
        if not os.path.isfile(self.status_path):
            return Status.NOT_STARTED
        with open(self.status_path, "r") as fp:
            status = fp.read().rstrip()
        Status.validate(status)
        return status

    def _set_status(self, value):
        Status.validate(value)
        path = self.status_path
        if value == Status.NOT_STARTED:
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path, "w") as fp:
            fp.write(f"{value}\n")

    def get_config(self) -> Dict[str, Any]:
        return {}

    @property
    def checkpoint(self) -> tf.train.Checkpoint:
        return tf.train.Checkpoint(root=self)

    def _restore(self, file_prefix: str):
        chkpt = self.checkpoint
        folder, _ = os.path.split(file_prefix)
        if not tf.io.gfile.isdir(folder) or not tf.io.gfile.exists(
            f"{file_prefix}.index"
        ):
            return None
        return chkpt.read(file_prefix)

    def restore_initial_state(self):
        return self._restore(self.initial_checkpoint_prefix)

    def restore_final_state(self):
        return self._restore(self.final_checkpoint_prefix)

    def run(self, callbacks: Iterable[cb.ExperimentCallback] = ()) -> Optional[T]:
        status = self.status
        if status == Status.FINISHED:
            logging.info("Experiment already finished.")
            return None

        callbacks = [
            *callbacks,
            cb.LoggingCallback(),
            # cb.SerializeCallback(self, self.serialized_path),
            cb.OperativeConfigLogger(self.operative_config_path),
        ]
        chkpt = self.checkpoint
        if chkpt is not None:
            callbacks.append(
                cb.Checkpoint(
                    chkpt, self.initial_checkpoint_prefix, self.final_checkpoint_prefix
                )
            )

        self._set_status(Status.RUNNING)
        for callback in callbacks:
            callback.on_start(status)

        try:
            results = self._run(status)
            self._set_status(Status.FINISHED)
            for callback in callbacks:
                callback.on_finished(results)
            return results
        except KeyboardInterrupt:
            self._set_status(Status.INTERRUPTED)
            for callback in callbacks:
                callback.on_interrupt()
            raise
        except Exception as exception:
            self._set_status(Status.EXCEPTION)
            for callback in callbacks:
                callback.on_exception(exception)
            raise

    @abc.abstractmethod
    def _run(self, start_status: str = Status.NOT_STARTED) -> T:
        """
        Run the experiment from the given starting status.

        Implementations do not need to save results.
        """
        raise NotImplementedError("Abstract method")


@gin.register(module="kb.experiment")
def run(experiment: Experiment):
    return experiment.run()
