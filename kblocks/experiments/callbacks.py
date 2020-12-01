import json
import os
import tempfile

import gin
from absl import logging

from kblocks.experiments import status as status_lib


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

    def on_finished(self):
        pass


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

    def on_start(self, status: status_lib.Status):
        if isinstance(status, status_lib.NotStarted):
            self.dump()


class LoggingCallback(ExperimentCallback):
    def on_start(self, status: status_lib.Status):
        if isinstance(status, status_lib.NotStarted):
            logging.info("Starting fresh experiment.")
        elif isinstance(status, status_lib.Running):
            logging.info("Continuing running experiment.")
        elif isinstance(status, status_lib.Excepted):
            logging.info("Restarting excepted experiment.")
        elif isinstance(status, status_lib.Interrupted):
            logging.info("Restarting interrupted experiment.")
        elif isinstance(status, status_lib.Finished):
            logging.info("Starting finished experiment.")
        else:
            raise ValueError("Invalid status.")

    def on_exception(self, exception: Exception):
        logging.info(f"Experiment failed with exception {exception}.")

    def on_interrupt(self):
        logging.info("Experiment interrupted.")

    def on_finished(self):
        logging.info("Experiment finished.")
