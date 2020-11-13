import tempfile

import tensorflow as tf

from kblocks.extras.callbacks.modules import variable_property


class Foo:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class FooModule(Foo, tf.Module):
    def __init__(self, a, b, name=None):
        tf.Module.__init__(self, name=name)
        Foo.__init__(self, a, b)

    a = variable_property("a", tf.int64)
    b = variable_property("b", tf.int64)


foo = FooModule(2, 3)
assert foo.a == 2
print(list(foo._flatten()))  # pylint: disable=protected-access

chkpt = tf.train.Checkpoint(foo=foo)
with tempfile.TemporaryDirectory() as tmp_dir:
    path = chkpt.save(tmp_dir)
    foo.a = 10
    chkpt.restore(path)
assert foo.a == 2
