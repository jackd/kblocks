# https://arxiv.org/abs/1904.03392
import gin
import tensorflow as tf


@gin.configurable(module="kb.layers")
class ChannelDropout(tf.keras.layers.Layer):
    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self._rate = rate

    @property
    def rate(self):
        return self._rate

    def get_config(self):
        config = super().get_config()
        config["rate"] = self.rate
        return config

    def __call__(self, features, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)

        assert isinstance(training, bool)
        if training:
            num_channels = features.shape[-1]
            mask = tf.random.uniform(shape=(num_channels,)) > self.rate
            return tf.where(mask, features / (1 - self.rate), tf.zeros_like(features))
        else:
            return features
