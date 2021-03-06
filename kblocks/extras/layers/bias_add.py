import gin
import tensorflow as tf

from kblocks.serialize import register_serializable


@gin.configurable(module="kb.extras.layers")
@register_serializable
class BiasAdd(tf.keras.layers.Layer):
    def __init__(
        self, initializer="zeros", regularizer=None, constraint=None, **kwargs
    ):
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)
        self.bias = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        s = input_shape[-1]
        self.bias = self.add_weight(
            "bias",
            shape=(s,),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
        )
        super().build(input_shape)

    def call(self, inputs):  # pylint: disable=arguments-differ
        return inputs + self.bias

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                initializer=tf.keras.initializers.serialize(self.initializer),
                regularizer=tf.keras.regularizers.serialize(self.regularizer),
                constraint=tf.keras.constraints.serialize(self.constraint),
            )
        )
        return config
