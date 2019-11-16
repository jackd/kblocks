from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
import gin
from typing import Sequence, Optional


@gin.configurable(module='kb')
class SessionOptions(object):

    def __init__(self,
                 allow_growth: bool = True,
                 visible_devices: Optional[Sequence[int]] = None):
        self.allow_growth = allow_growth
        self.visible_devices = visible_devices

    def configure_session(self):
        if self.visible_devices is not None:
            devices = tf.config.experimental.get_visible_devices('GPU')
            devices = {d.name.split(':')[-1]: d for d in devices}
            devices = [devices[str(d)] for d in self.visible_devices]
            tf.config.experimental.set_visible_devices(devices,
                                                       device_type='GPU')
        try:
            for device in tf.config.experimental.get_visible_devices('GPU'):
                tf.config.experimental.set_memory_growth(
                    device, self.allow_growth)
        except Exception:
            logging.info('Failed to set memory growth to {}'.format(
                self.allow_growth))

    def get_config(self):
        return dict(allow_growth=self.allow_growth,
                    visible_devices=self.visible_devices)

    @classmethod
    def from_config(cls, config):
        return SessionOptions(**config)
