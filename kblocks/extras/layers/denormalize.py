from __future__ import absolute_import, division, print_function

from typing import Optional

import gin
import tensorflow as tf

from kblocks.extras.layers.bias_add import BiasAdd
from kblocks.extras.layers.scale import Scale


@gin.configurable(module="kb.extras.layers")
class Denormalization(tf.keras.layers.Layer):
    def __init__(
        self, bias: Optional[BiasAdd] = None, scale: Optional[Scale] = None, **kwargs
    ):
        super(Denormalization, self).__init__(**kwargs)
        if bias is None:
            bias = BiasAdd()
        elif isinstance(scale, dict):
            bias = BiasAdd.from_config(bias)
        if scale is None:
            scale = Scale()
        elif isinstance(scale, dict):
            scale = Scale.from_config(scale)
        assert isinstance(scale, Scale)
        assert isinstance(bias, BiasAdd)
        self.bias = bias
        self.scale = scale

    def build(self, input_shape):
        if self.built:
            return
        self.bias.build(input_shape)
        self.scale.build(input_shape)

    def call(self, x):
        return self.bias(self.scale(x))

    def get_config(self):
        config = super(Denormalization, self).get_config()
        config["scale"] = self.scale.get_config()
        config["bias"] = self.bias.get_config()
        return config


# def Denormalization(bias: Optional[BiasAdd] = None,
#                     scale: Optional[Scale] = None):
#     if bias is None:
#         bias = BiasAdd()
#     if scale is None:
#         scale = Scale()
#     return tf.keras.models.Sequential([scale, bias])
