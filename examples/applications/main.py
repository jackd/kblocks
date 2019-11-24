"""Demonstrates usage of `kblocks.keras.applications`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kblocks.keras.applications import MobileNet
from gin import config

config.bind_parameter('tf.keras.layers.BatchNormalization.momentum', 0.9)

model = MobileNet((224, 224, 3))
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print('momentum = {}, {}'.format(layer.momentum, layer.name))
