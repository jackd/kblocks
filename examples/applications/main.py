"""Demonstrates usage of `kblocks.keras.applications`."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from gin import config

from kblocks.keras.applications import MobileNet

config.bind_parameter("tf.keras.layers.BatchNormalization.momentum", 0.9)

model = MobileNet((224, 224, 3))
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print("momentum = {}, {}".format(layer.momentum, layer.name))
