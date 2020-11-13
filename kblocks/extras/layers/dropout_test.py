import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from kblocks.extras.layers import dropout


class DropoutTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(dropout.Dropout, dropout.ChannelDropout)
    def test_test_mode(self, impl):
        shape = (1024, 8)
        rng = tf.random.Generator.from_seed(123)
        inputs = rng.uniform(shape)
        layer = impl(rate=0.2, seed=0)
        x = layer(inputs, training=False)
        np.testing.assert_equal(*self.evaluate((inputs, x)))

    @parameterized.parameters(
        (dropout.Dropout, 0.3),
        (dropout.Dropout, 0.5),
        (dropout.Dropout, 0.9),
        (dropout.ChannelDropout, 0.3),
        (dropout.ChannelDropout, 0.5),
        (dropout.ChannelDropout, 0.9),
    )
    def test_dropout_rate_mean(self, impl, rate):
        shape = (16, 16, 2048)
        rng = tf.random.Generator.from_seed(123)
        inputs = rng.uniform(shape)
        layer = impl(rate=rate, seed=0)
        x = layer(inputs, training=True)
        nnz = tf.math.count_nonzero(x)
        actual_rate = 1.0 - nnz / np.prod(shape)
        actual_mean = tf.reduce_mean(x)
        expected_mean = tf.reduce_mean(inputs)

        actual_mean, expected_mean, actual_rate = self.evaluate(
            (actual_mean, expected_mean, actual_rate)
        )
        np.testing.assert_allclose(actual_rate, rate, atol=0.01)
        np.testing.assert_allclose(actual_mean, expected_mean, atol=0.025)


if __name__ == "__main__":
    tf.test.main()
