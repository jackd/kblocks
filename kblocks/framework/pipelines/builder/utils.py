from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def assert_is_tensor_spec(spec, name='tensor_spec'):
    if not isinstance(spec, tf.TensorSpec):
        raise ValueError('{} must be a TensorSpec, got {}'.format(name, spec))
