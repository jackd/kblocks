import json
import os
import tempfile

import gin
import tensorflow as tf
from absl import logging

from kblocks.experiments.status import Status


def _sanitized(serialized):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "serialized.json")
        with open(path, "w") as fp:
            json.dump(serialized, fp)
        with open(path, "r") as fp:
            loaded = json.load(fp)
    return loaded


class ExperimentCallback:
    def on_start(self, status: str):
        pass

    def on_exception(self, exception: Exception):
        pass

    def on_interrupt(self):
        pass

    def on_finished(self, result):
        pass


class SerializeCallback(ExperimentCallback):
    def __init__(self, serializable, path: str):
        self._serializable = serializable
        self._path = path

    def dump(self):
        """Dump serialized data to disk."""
        path = self._path
        if os.path.exists(path):
            raise IOError(f"Serialized data already exists at {path}")
        with open(path, "w") as fp:
            json.dump(
                tf.keras.utils.serialize_keras_object(self._serializable),
                fp,
                indent=4,
                sort_keys=True,
            )

    def validate(self):
        """Ensure serialized data is consistent with what is already saved."""
        path = self._path
        if not os.path.exists(path):
            raise IOError(f"No serialized data saved at {path}")
        with open(path, "r") as fp:
            saved_serialized = json.load(fp)
        actual_serialized = tf.keras.utils.serialize_keras_object(self._serializable)
        actual_serialized = _sanitized(actual_serialized)
        if saved_serialized != actual_serialized:
            raise IOError(
                f"Saved serialized experiment different to current one at {path}.\n"
                f"Saved:\n{saved_serialized}\nCurrent:\n{actual_serialized}"
            )

    def on_start(self, status: str):
        if status == Status.NOT_STARTED:
            self.dump()
        else:
            self.validate()


class OperativeConfigLogger(ExperimentCallback):
    def __init__(self, path):
        self._path = path

    def dump(self):
        """Save `gin.operative_config_str()` to file."""
        exists = os.path.exists(self._path)
        if exists:
            raise IOError(f"Operative config already exists at {self._path}")
        with open(self._path, "w") as fp:
            fp.write(gin.operative_config_str())

    def on_start(self, status: str):
        if status == Status.NOT_STARTED:
            self.dump()


class LoggingCallback(ExperimentCallback):
    def on_start(self, status: str):
        if status == Status.NOT_STARTED:
            logging.info("Starting fresh experiment.")
        elif status == Status.RUNNING:
            logging.info("Continuing running experiment.")
        elif status == Status.EXCEPTION:
            logging.info("Restarting failed experiment.")
        elif status == Status.INTERRUPTED:
            logging.info("Restarting interrupted experiment.")
        elif status == Status.FINISHED:
            logging.info("Starting finished experiment.")
        else:
            raise ValueError("Invalid status.")

    def on_exception(self, exception: Exception):
        logging.info(f"Experiment failed with exception {exception}.")

    def on_interrupt(self):
        logging.info("Experiment interrupted.")

    def on_finished(self, result):
        logging.info("Experiment finished.")


class Checkpoint(ExperimentCallback):
    def __init__(
        self, checkpoint: tf.train.Checkpoint, initial_prefix: str, final_prefix: str
    ):
        self._checkpoint = checkpoint
        self._initial_prefix = initial_prefix
        self._final_prefix = final_prefix

    def _write(self, file_prefix: str):
        folder, _ = os.path.split(file_prefix)
        if not tf.io.gfile.exists(folder):
            tf.io.gfile.makedirs(folder)
        self._checkpoint.write(file_prefix)

    def on_start(self, status: str):
        if status == Status.NOT_STARTED:
            self._write(self._initial_prefix)

    def on_finished(self, result):
        self._write(self._final_prefix)
