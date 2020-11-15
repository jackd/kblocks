from typing import Optional

import gin
import tensorflow as tf

from kblocks.extras.layers.bias_add import BiasAdd
from kblocks.extras.layers.scale import Scale
from kblocks.serialize import register_serializable


@gin.configurable(module="kb.extras.layers")
@register_serializable
class Denormalization(tf.keras.layers.Layer):
    def __init__(
        self, bias: Optional[BiasAdd] = None, scale: Optional[Scale] = None, **kwargs
    ):
        super().__init__(**kwargs)
        if bias is None:
            bias = BiasAdd()
        elif isinstance(bias, dict):
            bias = BiasAdd.from_config(bias)
        assert isinstance(bias, BiasAdd)
        self.bias = bias

        if scale is None:
            scale = Scale()
        elif isinstance(scale, dict):
            scale = Scale.from_config(scale)
        assert isinstance(scale, Scale)
        self.scale = scale

    def build(self, input_shape):
        if self.built:
            return
        self.bias.build(input_shape)
        self.scale.build(input_shape)

    def call(self, x):  # pylint: disable=arguments-differ
        return self.bias(self.scale(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"scale": self.scale.get_config(), "bias": self.bias.get_config()}
        )
        return config
