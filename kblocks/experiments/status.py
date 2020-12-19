from typing import Union

import tensorflow as tf


class Status:
    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NotStarted(Status):
    pass


class Interrupted(Status):
    pass


class Finished(Status):
    pass


class Excepted(Status):
    def __init__(self, exception: Union[str, Exception]):
        self._exception = exception

    @property
    def exception(self):
        return self._exception

    def get_config(self):
        return dict(exception=str(self._exception))


class Running(Status):
    def __init__(self, pid: int):
        self._pid = pid

    @property
    def pid(self):
        return self._pid

    def get_config(self):
        return dict(pid=self._pid)


_module_objects = dict(locals())
del _module_objects["Union"]
del _module_objects["tf"]


def get(identifier: Union[str, Status]) -> Status:
    if isinstance(identifier, Status):
        return identifier
    status = tf.keras.utils.deserialize_keras_object(
        identifier, module_objects=_module_objects, printable_module_name="Status"
    )
    if not isinstance(status, Status):
        raise ValueError(f"Invalid status: expected Status instance, got {status}")
    return status
