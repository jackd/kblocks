"""
Demonstrates usage of `kblocks.keras.applications`.

Requires keras-applications - `pip install keras-applications`
"""


import tensorflow as tf
from gin import config

from kblocks.keras.applications import MobileNet

# set default batch-norm momentum to 0.9
config.bind_parameter("tf.keras.layers.BatchNormalization.momentum", 0.9)

model = MobileNet((224, 224, 3))
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print("momentum = {}, {}".format(layer.momentum, layer.name))
