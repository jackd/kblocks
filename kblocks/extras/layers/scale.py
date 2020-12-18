import gin
import tensorflow as tf

from kblocks.serialize import register_serializable


@gin.configurable(module="kb.extras.layers")
@register_serializable
class Scale(tf.keras.layers.Layer):
    def __init__(self, initializer="ones", regularizer=None, constraint=None, **kwargs):
        self.initializer = tf.keras.initializers.get(initializer)
        self.constraint = tf.keras.constraints.get(constraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.scalar = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                initializer=tf.keras.initializers.serialize(self.initializer),
                constraint=tf.keras.constraints.serialize(self.constraint),
                regularizer=tf.keras.regularizers.serialize(self.regularizer),
            )
        )
        return config

    def _scale_shape(self, input_shape):
        del self
        return input_shape[-1],

    def build(self, input_shape):
        if self.built:
            return
        self.scalar = self.add_weight(
            "scalar",
            shape=self._scale_shape(input_shape),
            dtype=self.dtype,
            initializer=self.initializer,
            constraint=self.constraint,
            regularizer=self.constraint,
        )
        super().build(input_shape)

    def call(self, inputs):  # pylint: disable=arguments-differ
        return inputs * self.scalar


@gin.configurable(module="kb.extras.layers")
@register_serializable
class UniformScale(Scale):
    def _scale_shape(self, input_shape):
        del self, input_shape
        return ()


@gin.configurable(module="kb.extras.layers")
def ZeroInit(initializer="zeros", regularizer=None, constraint=None, **kwargs):
    """
    Just a UniformScale with different default initializer.

    https://openreview.net/forum?id=BJeVklHtPr&fbclid=IwAR38PoAbf6WqlyLzMNuQURUJntGITfZvFFVb3z1a4Fiumwe1IM3lmjHGsl4
    """
    return UniformScale(
        initializer=initializer,
        regularizer=regularizer,
        constraint=constraint,
        **kwargs
    )
