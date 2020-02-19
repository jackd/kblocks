from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from kblocks.framework.problems.pipelines.builder.model_builder import ModelBuilder, UnbatchedModelBuilder


class ModelBuilderTest(tf.test.TestCase):

    def basic_test(self):
        b = ModelBuilder()
        inp = b.add_input(tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
        out = inp * 2
        b.add_output(out)
        model = b.build()

        x = tf.random.uniform((10, 3), dtype=tf.float32)
        actual = ModelBuilder.apply(model, x)
        expected = x * 2
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected)

    def unbatched_model_builder_test(self):
        b = UnbatchedModelBuilder()
        inp = b.add_input(tf.TensorSpec(shape=(), dtype=tf.float32))
        out = inp * 2
        b.add_output(out)
        model = b.build()

        x = tf.random.uniform((), dtype=tf.float32)
        model = b.build()
        actual = UnbatchedModelBuilder.apply(model, x)
        expected = x * 2
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    tf.test.main()
