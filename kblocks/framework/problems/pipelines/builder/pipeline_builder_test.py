from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from kblocks.framework.problems.pipelines.builder import pipeline_builder as pl_lib
from kblocks.framework.problems.pipelines.builder import PipelineModels as mod
from kblocks.extras.layers import ragged as ragged_layers


class Scaler(tf.keras.layers.Layer):

    def __init__(self, initializer, **kwargs):
        self._initializer = initializer
        super(Scaler, self).__init__(**kwargs)

    def build(self, input_shape):
        self._tensor = self.add_weight('scalar',
                                       initializer=self._initializer,
                                       shape=(),
                                       dtype=tf.float32)

    def call(self, inputs):
        return inputs * self._tensor


class PipelineBuilderTest(tf.test.TestCase):

    def test_single_io(self):
        batch_size = 2
        num_elements = 5
        scale_factor = 5.
        plb = pl_lib.PipelineBuilder(batch_size=batch_size)
        x_np = np.reshape(np.arange(batch_size * num_elements),
                          (batch_size, num_elements)).astype(np.float32)

        def gen():
            return x_np

        def pre_batch_map(x):
            return 2 * x

        def post_batch_map(x):
            return 3 * x

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (5,))

        inp = tf.nest.map_structure(plb.base_input, dataset.element_spec)
        self.assertEqual(inp.shape, dataset.element_spec.shape)
        self.assertEqual(inp.dtype, dataset.element_spec.dtype)

        batched = post_batch_map(plb.batch(pre_batch_map(inp)))
        trained = plb.trained_input(batched)
        plb.trained_output(
            Scaler(tf.keras.initializers.constant(scale_factor))(trained))

        pipeline, model = plb.build()
        dataset = pipeline(dataset)

        expected_output = post_batch_map(pre_batch_map(x_np)) * scale_factor

        for example in dataset:
            output = model(example)
            np.testing.assert_allclose(self.evaluate(output), expected_output)
            break

    def test_single_io_cached(self):
        batch_size = 2
        num_elements = 5
        scale_factor = 5.
        plb = pl_lib.PipelineBuilder(batch_size=batch_size, use_cache=True)
        x_np = np.reshape(np.arange(batch_size * num_elements),
                          (batch_size, num_elements)).astype(np.float32)

        def gen():
            return x_np

        def pre_cache_map(x):
            return 7 * x

        def pre_batch_map(x):
            return 2 * x

        def post_batch_map(x):
            return 3 * x

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (5,))

        inp = tf.nest.map_structure(plb.base_input, dataset.element_spec)
        self.assertEqual(inp.shape, dataset.element_spec.shape)
        self.assertEqual(inp.dtype, dataset.element_spec.dtype)

        batched = post_batch_map(
            plb.batch(pre_batch_map(plb.cache(pre_cache_map(inp)))))
        trained = plb.trained_input(batched)
        plb.trained_output(
            Scaler(tf.keras.initializers.constant(scale_factor))(trained))

        pipeline, model = plb.build()
        dataset = pipeline(dataset)

        expected_output = post_batch_map(pre_batch_map(
            pre_cache_map(x_np))) * scale_factor

        for example in dataset:
            output = model(example)
            np.testing.assert_allclose(self.evaluate(output), expected_output)
            break

    def test_ragged(self):
        batch_size = 2
        plb = pl_lib.PipelineBuilder(batch_size)
        x_np = [
            np.expand_dims(np.arange(5).astype(np.float32), axis=-1),
            np.expand_dims(np.arange(5, 12).astype(np.float32), axis=-1),
        ]
        scale_factor = 5.

        def gen():
            return x_np

        def pre_batch_map(x):
            if isinstance(x, list):
                return [xi * 2 for xi in x]
            elif isinstance(x, tf.RaggedTensor):
                return tf.ragged.map_flat_values(lambda x: x * 2, x)
            else:
                return x * 2

        def post_batch_map(x):
            if isinstance(x, list):
                return [xi * 3 for xi in x]
            else:
                return x * 3

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (None, 1))

        inp = plb.base_input(dataset.element_spec)
        self.assertEqual(inp.shape[0], dataset.element_spec.shape[0])
        self.assertEqual(inp.dtype, dataset.element_spec.dtype)

        x = tf.keras.layers.Lambda(pre_batch_map)(inp)
        batched = tf.keras.layers.Lambda(post_batch_map)(plb.batch(x,
                                                                   ragged=True))
        trained = plb.trained_input(batched)
        trained = Scaler(tf.keras.initializers.constant(scale_factor))(trained)
        plb.trained_output(ragged_layers.flat_values(trained))

        pipeline, model = plb.build()
        dataset = pipeline(dataset)

        expected_batched = post_batch_map([pre_batch_map(x) for x in gen()])
        expected_output = [scale_factor * x for x in expected_batched]
        expected_output = np.concatenate(expected_output, axis=0)

        for example in dataset:
            output = model(example)
            np.testing.assert_allclose(self.evaluate(output), expected_output)
            break

    def test_marks(self):
        batch_size = 2
        plb = pl_lib.PipelineBuilder(batch_size=batch_size)
        num_elements = 5
        x_np = np.reshape(np.arange(batch_size * num_elements),
                          (batch_size, num_elements)).astype(np.float32)
        scale_factor = 5.

        def gen():
            return x_np

        def pre_batch_map(x):
            return 2 * x

        def post_batch_map(x):
            return 3 * x

        dataset = tf.data.Dataset.from_generator(gen, tf.float32, (5,))
        inp = plb.base_input(dataset.element_spec)
        self.assertEqual(inp.shape, dataset.element_spec.shape)
        self.assertEqual(inp.dtype, dataset.element_spec.dtype)

        batched = post_batch_map(plb.batch(pre_batch_map(inp)))
        trained = plb.trained_input(batched)
        trained_out = Scaler(
            tf.keras.initializers.constant(scale_factor))(trained)
        plb.trained_output(trained_out)
        plb.get_mark(trained_out)

        self.assertEqual(plb.get_mark(inp), mod.PRE_BATCH)
        self.assertEqual(plb.get_mark(batched), mod.POST_BATCH)
        self.assertEqual(plb.get_mark(trained), mod.TRAINED)
        self.assertEqual(plb.get_mark(trained_out), mod.TRAINED)


if __name__ == '__main__':
    tf.test.main()
    # PipelineBuilderTest().test_single_io()
    # PipelineBuilderTest().test_marks()
    # PipelineBuilderTest().test_ragged()
