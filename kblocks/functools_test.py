import functools

import tensorflow as tf

from kblocks.functools import Function, get, partial, serialize_function


def f(x, y):
    return x + y


class FunctoolsTest(tf.test.TestCase):
    def test_function_wrapper(self):
        wrapped = Function(f)
        self.assertEqual(wrapped(4, 3), 7)
        self.assertEqual(partial(f, y=3)(4), 7)

    def test_serialized(self):
        serialized = serialize_function(f)
        self.assertEqual(
            serialized,
            dict(class_name="KBlocks>Function", config=dict(name="f", module=__name__)),
        )
        serialized = serialize_function(functools.partial(f, y=3))
        self.assertEqual(
            serialized,
            dict(
                class_name="functools>partial",
                config=dict(
                    func=dict(
                        class_name="KBlocks>Function",
                        config=dict(name="f", module=__name__),
                    ),
                    args=[],
                    keywords=dict(y=3),
                ),
            ),
        )

    def test_get(self):
        fn = get(
            dict(class_name="KBlocks>Function", config=dict(name="f", module=__name__))
        )
        self.assertEqual(fn(4, 3), 7)
        fn = get(
            dict(
                class_name="functools>partial",
                config=dict(
                    func=dict(
                        class_name="KBlocks>Function",
                        config=dict(name="f", module=__name__),
                    ),
                    args=[],
                    keywords=dict(y=3),
                ),
            )
        )
        self.assertEqual(fn(4), 7)


if __name__ == "__main__":
    tf.test.main()
